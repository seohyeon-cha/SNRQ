import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant
import pickle
import os
import glob
import re

from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
from algorithms.gptq import GPTQ
from algorithms.gptaq import GPTAQ
from algorithms.greedyaq import GreedyAQ
from algorithms.foem import FOEM
from algorithms.ldlq import LDLQ
from algorithms.guidedquant import GuidedQuant
from algorithms.gptq import Observer  # Observer is the same across all algorithms
from utils import find_layers, DEV, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
import copy
import transformers
import utils
from utils.plot_delta_x import save_alpha_trace_plot, save_alpha_per_module_plot, plot_delta_x_2d_3d, compute_and_save_layer_norms, plot_unified_mae, load_mae_from_pickle

def get_llama(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_sequential(model, dataloader, dev, fp_path):

    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    # model = model.cuda()
    layers[0] = Catcher(layers[0].cuda())
    for batch in dataloader:
        try:
            model(batch[0].to(dev).cuda())
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    inps_ = inps.data.clone()
    outs_ = outs.data.clone()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    begin_time = time.time()

    # Store transformer block output dX for plotting
    transformer_block_dx = [] if getattr(args, 'plot_delta_x', False) else None  # per-layer per-channel MAE
    layer_mae = []   # scalar MAE per layer
    layer_fro = []   # scalar Frobenius norm per layer

    sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]

    if args.method in ["gptaq", "greedyaq"]:
        fp_inputs_cache = utils.modelutils.FPInputsCache(sequential)
        fp_inps = inps.clone()

    args.alpha_track = []
    args.alpha_per_module = {}

    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        if args.method in ["gptaq", "greedyaq"]:
            fp_inputs_cache.add_hook(full)

            for j in range(args.nsamples):
                fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            fp_inputs_cache.clear_hook()


        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}

            for name in subset:
                if args.method == "gptaq":
                    gptq[name] = GPTAQ(subset[name], observe=args.observe)
                elif args.method == "greedyaq":
                    if args.alpha_method == "sample":
                        gptq[name] = GreedyAQ(subset[name], observe=args.observe, 
                                              sampled_alpha=True, mixup_param=args.mixup_param, seed=args.seed)
                    else:
                        gptq[name] = GreedyAQ(subset[name], observe=args.observe)
                elif args.method == "foem":
                    gptq[name] = FOEM(subset[name], observe=args.observe)
                elif args.method == "gptq":
                    gptq[name] = GPTQ(subset[name], observe=args.observe)
                elif args.method == "ldlq":
                    gptq[name] = LDLQ(subset[name], observe=args.observe)
                elif args.method == "guidedquant":
                    # GuidedQuant can load pre-computed Hessians from cache
                    hessian_path = getattr(args, 'hessian_path', None)
                    gptq[name] = GuidedQuant(subset[name], observe=args.observe, hessian_path=hessian_path)
                else:
                    raise ValueError(f"Method {args.method} not supported.")
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                # Only GPTAQ and GreedyAQ need fp_inp
                if args.method in ["gptaq", "greedyaq"]:
                    gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            first_module_name = list(subset.keys())[0]
            handle = subset[first_module_name].register_forward_hook(add_batch(first_module_name))

            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            
            handle.remove()

            # copy H and dXXT
            for name in subset:
                if name != first_module_name:
                    # All methods share Hessian H across modules in the same sequential group
                    gptq[name].H = gptq[first_module_name].H
                    if args.method in ["gptaq", "greedyaq"]:
                        gptq[name].dXXT = gptq[first_module_name].dXXT
                        if hasattr(gptq[first_module_name], 'dXdXT'):
                            gptq[name].dXdXT = gptq[first_module_name].dXdXT
                  
            for name in subset:
                # GreedyAQ requires args parameter, others don't need it
                if args.method == "greedyaq":
                    if args.alpha_method == "optimize":
                        if i == 0:
                            args.alpha_per_module[name] = [0.0]
                        if i == 1:
                            if args.groupsize == -1 or args.model == "meta-llama/Llama-2-70b-hf":
                                alpha_ref = 0.25
                            else:
                                alpha_ref = 0.5 
                            args.alpha_per_module[name].append(alpha_ref) 
                        
                        args.alpha = args.alpha_per_module[name][-1]
                    
                    if args.alpha_method == "alternate":
                        scale, zero, g_idx, error = gptq[name].fasterquant_alternating_method(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name, alpha=args.alpha, args=args)
                    else:
                        scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name, alpha=args.alpha, beta=args.beta, args=args)
                else:
                    scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name, alpha=args.alpha, beta=args.beta, args=args)
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()


        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        if getattr(args, 'plot_delta_x', False) and args.method in ["gptaq", "greedyaq"]:
            # X_f = fp_inps, X_q = outs
            dx_block = fp_inps - outs  # [nsamples, seqlen, hidden_size]

            # Flatten over batch & time, keep hidden dimension as "channels"
            # shape: [hidden_size, nsamples * seqlen]
            dx_block_reshaped = dx_block.permute(2, 0, 1).reshape(dx_block.shape[2], -1)

            # ---- scalar diagnostics per layer ----
            # MAE over all positions & channels
            mae_layer = dx_block_reshaped.abs().mean().item()
            layer_mae.append(mae_layer)

            # Frobenius norm over all positions & channels
            # (linalg.norm with default ord=2 on 2D is Frobenius)
            fro_layer = torch.linalg.norm(dx_block_reshaped).item()
            layer_fro.append(fro_layer)

            # ---- per-channel MAE for 3D plot ----
            # average |deltaX| for each hidden channel
            mae_per_channel = dx_block_reshaped.abs().mean(dim=1).cpu()  # [hidden_size]
            transformer_block_dx.append(mae_per_channel)

        if args.method in ["gptaq", "greedyaq"]:
            fp_inputs_cache.clear_cache()
        
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        inps_, outs_ = outs_, inps_
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    end_time = time.time()

    print(f'time cost: {end_time-begin_time}s')
    
    # Plot transformer block output dX if enabled
    if getattr(args, 'plot_delta_x', False) and len(transformer_block_dx) > 0:
        plot_path_2d = f"{args.plot_delta_x_path}/transformer_block_dx_2d_{args.method}.png"
        plot_path_3d = f"{args.plot_delta_x_path}/transformer_block_dx_3d_{args.method}.png"

        # transformer_block_dx: list of [hidden_size] per layer (per-channel MAE)
        # layer_mae: list of scalar MAE per layer
        # layer_fro: list of scalar Frobenius norms per layer
        plot_delta_x_2d_3d(
            transformer_block_dx,   # per-channel MAE per layer
            layer_mae,              # scalar MAE per layer
            layer_fro,              # Fro norm per layer
            method_name=args.method,
            output_path_2d=plot_path_2d,
            output_path_3d=plot_path_3d
        )
        
        # Save MAE and Frobenius norms per layer (for unified plot across alphas)
        alpha_val = getattr(args, 'alpha', None)
        alpha_method = getattr(args, 'alpha_method', 'fixed')
        mixup_param = getattr(args, 'mixup_param', None)
        
        # Determine filename based on alpha method
        if alpha_method == 'fixed' and alpha_val is not None:
            # Fixed alpha: use alpha value
            norm_save_path = f"{args.plot_delta_x_path}/layer_norms_alpha{alpha_val}.pkl"
            compute_and_save_layer_norms(layer_fro, norm_save_path, layer_mae_list=layer_mae)
            
            calibration_mae_path = f"{args.plot_delta_x_path}/calibration_mae_alpha{alpha_val}.pkl"
            os.makedirs(os.path.dirname(calibration_mae_path), exist_ok=True)
            with open(calibration_mae_path, 'wb') as f:
                pickle.dump({
                    'layer_mae': layer_mae,
                    'alpha': alpha_val,
                    'alpha_method': alpha_method,
                    'method': args.method,
                    'dataset': 'calibration'
                }, f)
            print(f'Saved calibration MAE to {calibration_mae_path}')
        elif alpha_method in ['sample', 'alternate'] and mixup_param is not None:
            # Sampled/alternate alpha: use mixup_param (beta)
            beta_tag = f"beta{mixup_param}".replace('.', 'p')
            calibration_mae_path = f"{args.plot_delta_x_path}/calibration_mae_{beta_tag}.pkl"
            os.makedirs(os.path.dirname(calibration_mae_path), exist_ok=True)
            with open(calibration_mae_path, 'wb') as f:
                pickle.dump({
                    'layer_mae': layer_mae,
                    'alpha': None,
                    'alpha_method': alpha_method,
                    'mixup_param': mixup_param,
                    'method': args.method,
                    'dataset': 'calibration'
                }, f)
            print(f'Saved calibration MAE to {calibration_mae_path}')
        else:
            # Fallback for other cases
            calibration_mae_path = f"{args.plot_delta_x_path}/calibration_mae_{args.method}.pkl"
            os.makedirs(os.path.dirname(calibration_mae_path), exist_ok=True)
            with open(calibration_mae_path, 'wb') as f:
                pickle.dump({
                    'layer_mae': layer_mae,
                    'alpha': alpha_val,
                    'alpha_method': alpha_method,
                    'method': args.method,
                    'dataset': 'calibration'
                }, f)
            print(f'Saved calibration MAE to {calibration_mae_path}')

    if len(args.alpha_track) > 0 and args.alpha_method == "sample":
        save_alpha_trace_plot(args, out_path=f"{args.plot_delta_x_path}/alpha_trace-{args.seed}.png")
        save_alpha_per_module_plot(args, out_path=f"{args.plot_delta_x_path}/alpha_per_module-{args.seed}.png")
    
    if args.observe:
        observer.print()
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in observer.items():
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            print('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                # GreedyAQ requires args parameter, others don't need it
                if args.method == "greedyaq":
                    scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name, alpha=args.alpha, beta=args.beta, args=args)
                else:
                    scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name, alpha=args.alpha, beta=args.beta)

                table.add_row([wbits, groupsize, error])
                quantizers['model.layers.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    # model = model.to(dev)
    layers[0] = Catcher(layers[0].to(dev))
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    
    return ppl.item()


@torch.no_grad()
def llama_eval_with_mae(quantized_model, fp_model, testenc, dev, method_name: str = ""):
    """
    Evaluate quantized model on validation set and collect MAE between FP and quantized activations.
    
    Args:
        quantized_model: The quantized model
        fp_model: The full precision model
        testenc: Test encoder/dataloader
        dev: Device
        method_name: Name of quantization method (for saving)
        
    Returns:
        layer_mae: List of MAE per layer
    """
    print('Evaluating with MAE collection on validation set...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // quantized_model.seqlen
    # Limit to reasonable number of samples for efficiency
    nsamples = min(nsamples, 32)  # Use up to 32 samples

    use_cache_q = quantized_model.config.use_cache
    use_cache_fp = fp_model.config.use_cache
    quantized_model.config.use_cache = False
    fp_model.config.use_cache = False
    
    layers_q = quantized_model.model.layers
    layers_fp = fp_model.model.layers

    # Setup for quantized model
    quantized_model.model.embed_tokens = quantized_model.model.embed_tokens.to(dev)
    layers_q[0] = layers_q[0].to(dev)
    
    # Setup for FP model
    fp_model.model.embed_tokens = fp_model.model.embed_tokens.to(dev)
    layers_fp[0] = layers_fp[0].to(dev)

    dtype = next(iter(quantized_model.parameters())).dtype
    inps_q = torch.zeros((nsamples, quantized_model.seqlen, quantized_model.config.hidden_size), dtype=dtype, device=dev)
    inps_fp = torch.zeros((nsamples, fp_model.seqlen, fp_model.config.hidden_size), dtype=dtype, device=dev)
    
    cache_q = {'i': 0, 'attention_mask': None, 'position_ids': None}
    cache_fp = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module, cache_dict):
            super().__init__()
            self.module = module
            self.cache_dict = cache_dict

        def forward(self, inp, **kwargs):
            inps = self.cache_dict['inps']
            inps[self.cache_dict['i']] = inp
            self.cache_dict['i'] += 1
            self.cache_dict['attention_mask'] = kwargs['attention_mask']
            self.cache_dict['position_ids'] = kwargs['position_ids']
            raise ValueError

    # Capture inputs for quantized model
    cache_q['inps'] = inps_q
    layers_q[0] = Catcher(layers_q[0].to(dev), cache_q)
    for i in range(nsamples):
        batch = testenc[:, (i * quantized_model.seqlen):((i + 1) * quantized_model.seqlen)].to(dev)
        try:
            quantized_model(batch.to(dev))
        except ValueError:
            pass
    layers_q[0] = layers_q[0].module
    
    # Capture inputs for FP model
    cache_fp['inps'] = inps_fp
    layers_fp[0] = Catcher(layers_fp[0].to(dev), cache_fp)
    for i in range(nsamples):
        batch = testenc[:, (i * fp_model.seqlen):((i + 1) * fp_model.seqlen)].to(dev)
        try:
            fp_model(batch.to(dev))
        except ValueError:
            pass
    layers_fp[0] = layers_fp[0].module

    layers_q[0] = layers_q[0].cpu()
    layers_fp[0] = layers_fp[0].cpu()
    quantized_model.model.embed_tokens = quantized_model.model.embed_tokens.cpu()
    fp_model.model.embed_tokens = fp_model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs_q = torch.zeros_like(inps_q)
    outs_fp = torch.zeros_like(inps_fp)
    attention_mask_q = cache_q['attention_mask']
    position_ids_q = cache_q['position_ids']
    attention_mask_fp = cache_fp['attention_mask']
    position_ids_fp = cache_fp['position_ids']

    layer_mae = []
    
    for i in range(len(layers_q)):
        layer_q = layers_q[i].to(dev)
        layer_fp = layers_fp[i].to(dev)

        # Run quantized layer
        for j in range(nsamples):
            outs_q[j] = layer_q(inps_q[j].unsqueeze(0), attention_mask=attention_mask_q, position_ids=position_ids_q)[0]
        
        # Run FP layer
        for j in range(nsamples):
            outs_fp[j] = layer_fp(inps_fp[j].unsqueeze(0), attention_mask=attention_mask_fp, position_ids=position_ids_fp)[0]
        
        # Compute MAE between FP and quantized outputs
        dx = outs_fp - outs_q  # [nsamples, seqlen, hidden_size]
        dx_reshaped = dx.permute(2, 0, 1).reshape(dx.shape[2], -1)  # [hidden_size, nsamples * seqlen]
        mae_layer = dx_reshaped.abs().mean().item()
        layer_mae.append(mae_layer)
        
        layers_q[i] = layer_q.cpu()
        layers_fp[i] = layer_fp.cpu()
        del layer_q, layer_fp
        torch.cuda.empty_cache()
        inps_q, outs_q = outs_q, inps_q
        inps_fp, outs_fp = outs_fp, inps_fp

    quantized_model.config.use_cache = use_cache_q
    fp_model.config.use_cache = use_cache_fp
    
    print(f'Collected MAE for {len(layer_mae)} layers')
    return layer_mae


# TODO: perform packing on GPU
def llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model

def llama_multigpu(model, gpus, gpu_dist):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    if hasattr(model.model, 'norm') and model.model.norm:
        model.model.norm = model.model.norm.to(gpus[0])
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='llama model to load')
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 5, 6, 7, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--test-generation', action='store_true', help='test generation.')
    parser.add_argument('--lm-eval', action='store_true', help='evaluate quantized model using lm_eval.')
    parser.add_argument('--lm-eval-batch-size', type=int, default=32, help='Batch size for lm_eval.')
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"],
        help='Tasks for lm_eval. Use format "task_name:num_fewshot" for few-shot tasks (e.g., "mmlu:5" for 5-shot MMLU).'   
    )
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
    parser.add_argument('--step', action='store_true', help='')
    parser.add_argument('--step_bits', type=int, default=8)
    parser.add_argument('--method', type=str, default='', help='Method to use for quantization.')
    parser.add_argument('--sort-asym', action='store_true', help='Whether to sort asymmetric quantization levels.')
    parser.add_argument('--alpha-method', type=str, default='fixed', choices=['optimize', 'fixed', 'alternate', 'sample'], help='Method to use for alpha update.')
    parser.add_argument('--mixup-param', type=float, default=0.0, help='Mixup parameter for GreedyAQ.')
    parser.add_argument('--alpha', type=float, default=0.25, help='Coefficient for weight correction term')
    parser.add_argument('--cd_passes', type=int, default=0, help='Number of coordinate descent passes for GreedyAQ.')
    # for beam search
    parser.add_argument('--beam-size', type=int, default=1, help='Coefficient for weight correction term')
    parser.add_argument('--beam-cands', type=int, default=3, help='Coefficient for weight correction term')
    parser.add_argument('--beam-sigma', type=float, default=0.25, help='Coefficient for weight correction term')
    parser.add_argument('--beam-k', type=int, default=16, help='Coefficient for weight correction term')
    parser.add_argument('--nn_beam', action='store_true', help='Whether to plot delta X values and generate 3D plots of |X_q - X_f|')

    parser.add_argument('--beta', type=float, default=0.0003, help='Coefficient for weight correction term')
    parser.add_argument('--incoh-process', action='store_true', help='Whether to perform incoherence process.')
    parser.add_argument('--incoh-mode', type=str, default='kron', choices=['had', 'kron'], help='Incoherence mode for GreedyAQ.')
    parser.add_argument('--rescale-WH', action='store_true', help='Whether to rescale W and H to minimize proxy loss.')
    parser.add_argument('--rescale-D', action='store_true', help='Whether to rescale W and H to minimize proxy loss.')
    parser.add_argument('--plot-delta-x', action='store_true', help='Whether to plot delta X values and generate 3D plots of |X_q - X_f|')
    parser.add_argument('--plot-delta-x-path', type=str, default='plots/delta_x', help='Directory path to save delta X plots')
    parser.add_argument('--eval-mae-validation', action='store_true', help='Collect MAE on C4 validation set and save for unified plotting')
    parser.add_argument('--plot-unified-mae', type=str, default='', help='Path to directory containing pickle files with MAE data to plot. Files should be named like "validation_mae_alpha{alpha}.pkl"')
    parser.add_argument('--ours', action='store_true', help='Use our method')
    parser.add_argument('--ours_v2', action='store_true', help='Use our method')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='llm-quantization', help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default='', help='Wandb run name (default: auto-generated)')
    parser.add_argument('--hessian-path', type=str, default='cache/hessians', help='Path to directory containing pre-computed Hessian files for GuidedQuant (default: cache/hessians)')

    args = parser.parse_args()


    if args.wandb:
        import wandb
        run_name = args.wandb_name if args.wandb_name else f"{args.method}_{args.wbits}bit_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                'model': args.model,
                'method': args.method,
                'wbits': args.wbits,
                'groupsize': args.groupsize,
                'seed': args.seed,
                'alpha': args.alpha,
                'alpha_method': args.alpha_method,
                'beta': args.beta,
                'act_order': args.act_order,
                'true_sequential': args.true_sequential,
                'incoh_process': args.incoh_process,
                'incoh_mode': args.incoh_mode,
                'rescale_D': args.rescale_D,
                'mixup_param': args.mixup_param,
                'beam_size': args.beam_size,
                'beam_cands': args.beam_cands, 
                'nn_beam': args.nn_beam,
                'cd_passes': args.cd_passes,

            }
        )

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load and not args.step:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_llama(args.model)
        model.eval()
        if args.step:
            print('load step!')
            model_high = load_quant(args.model, args.load, args.step_bits, args.groupsize)
            model_high.eval()
            for name, param in model.named_parameters():
                print(name)
                param.data = model_high.get_buffer('.'.join(name.split('.')[:-1]+['qweight']))

    dataloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen)

    quantizers = {}  # Initialize quantizers dict
    if (not args.load and args.wbits < 16 and not args.nearest) or args.step:
        # Default to gptq if method not specified

        torch.cuda.synchronize()
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args.model)
        torch.cuda.synchronize()
        quant_time = time.time() - tick
        print(f"Quantization time: {quant_time}s")
        if args.wandb:
            wandb.log({'quantization_time': quant_time})

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        # datasets = ['wikitext2', 'ptb', 'c4']
        datasets = ['wikitext2', 'c4']
        if args.new_eval:
            datasets = ['wikitext2', 'c4-new']
        eval_results = {}
        for dataset in datasets:
            testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, eval_mode=True)
            print(dataset)
            ppl = llama_eval(model, testloader, DEV)
            eval_results[f'{dataset}_perplexity'] = ppl
        
        if args.wandb:
            # Log eval results
            wandb.log(eval_results)
    
    # Collect MAE on C4 validation set if requested
    if args.eval_mae_validation:
        print('Collecting MAE on C4 validation set...')
        # Load full precision model for comparison
        fp_model = get_llama(args.model)
        fp_model.eval()
        
        # Get C4 validation set
        c4_testloader = get_loaders('c4', seed=args.seed, model=args.model, seqlen=model.seqlen, eval_mode=True)
        
        # Collect MAE
        validation_layer_mae = llama_eval_with_mae(model, fp_model, c4_testloader, DEV, method_name=args.method)
        
        # Save MAE data to pickle file
        alpha_val = getattr(args, 'alpha', None)
        alpha_method = getattr(args, 'alpha_method', 'fixed')
        mixup_param = getattr(args, 'mixup_param', None)
        
        # Determine filename based on alpha method
        if alpha_method == 'fixed' and alpha_val is not None:
            mae_save_path = f"{args.plot_delta_x_path}/validation_mae_alpha{alpha_val}.pkl"
            save_data = {
                'layer_mae': validation_layer_mae,
                'alpha': alpha_val,
                'alpha_method': alpha_method,
                'method': args.method
            }
        elif alpha_method in ['sample', 'alternate'] and mixup_param is not None:
            beta_tag = f"beta{mixup_param}".replace('.', 'p')
            mae_save_path = f"{args.plot_delta_x_path}/validation_mae_{beta_tag}.pkl"
            save_data = {
                'layer_mae': validation_layer_mae,
                'alpha': None,
                'alpha_method': alpha_method,
                'mixup_param': mixup_param,
                'method': args.method
            }
        else:
            mae_save_path = f"{args.plot_delta_x_path}/validation_mae_{args.method}.pkl"
            save_data = {
                'layer_mae': validation_layer_mae,
                'alpha': alpha_val,
                'alpha_method': alpha_method,
                'method': args.method
            }
        
        os.makedirs(os.path.dirname(mae_save_path), exist_ok=True)
        with open(mae_save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'Saved validation MAE to {mae_save_path}')
        
        del fp_model
        torch.cuda.empty_cache()
    
    # Plot unified MAE across different alphas if requested
    if args.plot_unified_mae:
        # Find all validation MAE pickle files (both fixed alpha and sampled beta)
        validation_alpha_pattern = os.path.join(args.plot_unified_mae, 'validation_mae_alpha*.pkl')
        validation_beta_pattern = os.path.join(args.plot_unified_mae, 'validation_mae_beta*.pkl')
        validation_files = glob.glob(validation_alpha_pattern) + glob.glob(validation_beta_pattern)
        
        # Find all calibration MAE pickle files (both fixed alpha and sampled beta)
        calibration_alpha_pattern = os.path.join(args.plot_unified_mae, 'calibration_mae_alpha*.pkl')
        calibration_beta_pattern = os.path.join(args.plot_unified_mae, 'calibration_mae_beta*.pkl')
        calibration_files = glob.glob(calibration_alpha_pattern) + glob.glob(calibration_beta_pattern)
        
        if not validation_files and not calibration_files:
            print(f'No MAE pickle files found in {args.plot_unified_mae}')
        else:
            validation_mae_dict = {}  # Will store label -> mae_list mapping
            calibration_mae_dict = {}  # Will store label -> mae_list mapping
            
            # Load validation MAE data
            for pickle_file in validation_files:
                try:
                    data = load_mae_from_pickle(pickle_file)
                    if 'layer_mae' not in data:
                        print(f'Warning: {pickle_file} does not contain layer_mae data')
                        continue
                    
                    # Determine label based on alpha_method
                    alpha_method = data.get('alpha_method', 'fixed')
                    if alpha_method == 'fixed' and 'alpha' in data and data['alpha'] is not None:
                        # Fixed alpha: use alpha value as label
                        alpha_val = data['alpha']
                        label = f'alpha={alpha_val}'
                        validation_mae_dict[label] = data['layer_mae']
                    elif alpha_method in ['sample', 'alternate'] and 'mixup_param' in data:
                        # Sampled/alternate: use beta distribution notation
                        beta_val = data['mixup_param']
                        label = f'β({beta_val}, {beta_val})'
                        validation_mae_dict[label] = data['layer_mae']
                    else:
                        # Fallback: use filename
                        basename = os.path.basename(pickle_file)
                        label = basename.replace('.pkl', '').replace('validation_mae_', '')
                        validation_mae_dict[label] = data['layer_mae']
                except (ValueError, KeyError) as e:
                    print(f'Warning: Could not process {pickle_file}: {e}')
            
            # Load calibration MAE data
            for pickle_file in calibration_files:
                try:
                    data = load_mae_from_pickle(pickle_file)
                    if 'layer_mae' not in data:
                        print(f'Warning: {pickle_file} does not contain layer_mae data')
                        continue
                    
                    # Determine label based on alpha_method
                    alpha_method = data.get('alpha_method', 'fixed')
                    if alpha_method == 'fixed' and 'alpha' in data and data['alpha'] is not None:
                        # Fixed alpha: use alpha value as label
                        alpha_val = data['alpha']
                        label = f'alpha={alpha_val}'
                        calibration_mae_dict[label] = data['layer_mae']
                    elif alpha_method in ['sample', 'alternate'] and 'mixup_param' in data:
                        # Sampled/alternate: use beta distribution notation
                        beta_val = data['mixup_param']
                        label = f'β({beta_val}, {beta_val})'
                        calibration_mae_dict[label] = data['layer_mae']
                    else:
                        # Fallback: use filename
                        basename = os.path.basename(pickle_file)
                        label = basename.replace('.pkl', '').replace('calibration_mae_', '')
                        calibration_mae_dict[label] = data['layer_mae']
                except (ValueError, KeyError) as e:
                    print(f'Warning: Could not process {pickle_file}: {e}')
            
            if validation_mae_dict or calibration_mae_dict:
                # Create separate plots for validation and calibration
                if validation_mae_dict:
                    validation_output_path = os.path.join(args.plot_unified_mae, 'unified_validation_mae.png')
                    plot_unified_mae(
                        validation_mae_dict,
                        validation_output_path,
                        title='MAE on C4 Validation Set',
                        calibration_data_dict=None
                    )
                    print(f'Saved validation MAE plot (with zoomed inset) to {validation_output_path}')
                
                if calibration_mae_dict:
                    calibration_output_path = os.path.join(args.plot_unified_mae, 'unified_calibration_mae.png')
                    plot_unified_mae(
                        calibration_mae_dict,
                        calibration_output_path,
                        title='MAE on C4 Calibration Set',
                        calibration_data_dict=None
                    )
                    print(f'Saved calibration MAE plot (with zoomed inset) to {calibration_output_path}')
            else:
                print('No valid MAE data found in pickle files')

    if args.test_generation:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        from transformers import LlamaTokenizer, TextStreamer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        input_ids = tokenizer(["The capital of New Mexico is"], return_tensors="pt").input_ids.to(gpus[0])
        streamer = TextStreamer(tokenizer)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, streamer=streamer)
        

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        # import pdb; pdb.set_trace()
        model.save_pretrained(f'ckpts/{args.save}')
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=getattr(args, 'hf_token', None))
        tokenizer.save_pretrained(f'ckpts/{args.save}')
        # llama_6(model, quantizers, args.wbits, args.groupsize)
        # torch.save(model.state_dict(), args.save)

    if args.lm_eval:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        model.to(DEV)

        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=getattr(args, 'hf_token', None))
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        task_names = args.tasks
        results = lm_eval.simple_evaluate(hflm, tasks=task_names, batch_size=args.lm_eval_batch_size)['results']

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        print(metric_vals)
        
        if args.wandb:
            wandb.log(metric_vals)
            
            # Create a table with run information and metrics
            # Each row represents one run
            table_data = []
            
            # Prepare row data: metadata first, then task metrics
            row = [
                args.method if args.method else 'fp16',
                args.wbits,
                args.seed,
                args.groupsize if hasattr(args, 'groupsize') else None,
            ]
            
            # Add task metrics in order
            for task in sorted(results.keys()):
                acc = round(results[task].get('acc_norm,none', results[task].get('acc,none', 0)), 4)
                row.append(acc)
            
            # Add average accuracy
            row.append(metric_vals['acc_avg'])
            
            table_data.append(row)
            
            # Define column names
            columns = ['method', 'wbits', 'seed', 'groupsize']
            columns.extend([f'{task}_acc' for task in sorted(results.keys())])
            columns.append('acc_avg')
            
            # Create and log the table
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({"lm_eval_results_table": table})


    if args.wandb:
        wandb.finish()

    if not args.observe and args.save_safetensors:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)
        