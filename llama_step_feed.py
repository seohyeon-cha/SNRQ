import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant
import os

from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
from algorithms.gptq import GPTQ
from algorithms.gptaq import GPTAQ
from algorithms.greedyaq import GreedyAQ
from algorithms.foem import FOEM
from algorithms.gptq import Observer  # Observer is the same across all algorithms
from utils import find_layers, DEV, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
import copy
import transformers
import utils
from utils import gradient_utils

def parse_layers_to_update(args, n_layers: int):

    # fallback: if "sensitive" is requested but you haven't implemented it here
    # use last-k as a safe default
    k = int(getattr(args, "n_layers_to_update", 0))
    return set(range(max(0, n_layers - k), n_layers))


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
def llama_sequential(
    student_model,
    dataloader,
    dev,
    teacher_model=None,
    args=None,
    kd_gradients=None,            # dict[layer_idx][module_name] -> grad tensor
    requantize_layers=None,       # set of layer indices to requantize in pass2
    mode="quantize",              # "quantize" (pass1) or "requantize" (pass2)
):
    assert mode in ["quantize", "requantize"]
    if args.method in ["gptaq", "greedyaq"]:
        assert teacher_model is not None, "teacher_model is required for gptaq/greedyaq to build fp_inp cache."

    use_cache = student_model.config.use_cache
    student_model.config.use_cache = False
    layers_s = student_model.model.layers
    n_layers = len(layers_s)

    # ---------- capture student inputs to layer0 ----------
    student_model.model.embed_tokens = student_model.model.embed_tokens.to(dev)
    student_model.model.norm = student_model.model.norm.to(dev)
    layers_s[0] = layers_s[0].to(dev)

    dtype = next(iter(student_model.parameters())).dtype
    inps = torch.zeros((args.nsamples, student_model.seqlen, student_model.config.hidden_size),
                       dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}


    class Catcher(nn.Module):
        def __init__(self, module): super().__init__(); self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers_s[0] = Catcher(layers_s[0])
    for batch in dataloader:
        try:
            student_model(batch[0].to(dev))
        except ValueError:
            pass
    layers_s[0] = layers_s[0].module

    layers_s[0] = layers_s[0].cpu()
    student_model.model.embed_tokens = student_model.model.embed_tokens.cpu()
    student_model.model.norm = student_model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    student_model.model.rotary_emb = student_model.model.rotary_emb.to(dev)
    _, seqlen = position_ids.shape
    dummy = torch.empty((1, seqlen, student_model.config.hidden_size), device=dev, dtype=inps.dtype)
    position_embeddings = student_model.model.rotary_emb(dummy, position_ids)


    # ---------- capture teacher inputs to layer0 (needed for fp_inp cache) ----------
    if args.method in ["gptaq", "greedyaq"]:
        teacher_model.config.use_cache = False
        layers_t = teacher_model.model.layers
        teacher_model.model.embed_tokens = teacher_model.model.embed_tokens.to(dev)
        teacher_model.model.norm = teacher_model.model.norm.to(dev)
        layers_t[0] = layers_t[0].to(dev)

        fp_inps = torch.zeros_like(inps)
        fp_cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

        class CatcherT(nn.Module):
            def __init__(self, module): super().__init__(); self.module = module
            def forward(self, inp, **kwargs):
                fp_inps[fp_cache['i']] = inp
                fp_cache['i'] += 1
                fp_cache['attention_mask'] = kwargs['attention_mask']
                fp_cache['position_ids'] = kwargs['position_ids']
                raise ValueError

        layers_t[0] = CatcherT(layers_t[0])
        for batch in dataloader:
            try:
                teacher_model(batch[0].to(dev))
            except ValueError:
                pass
        layers_t[0] = layers_t[0].module

        layers_t[0] = layers_t[0].cpu()
        teacher_model.model.embed_tokens = teacher_model.model.embed_tokens.cpu()
        teacher_model.model.norm = teacher_model.model.norm.cpu()
        torch.cuda.empty_cache()

        # sanity: masks/pos match
        # (they should, since same tokenization / seqlen)
        fp_inputs_cache = utils.modelutils.FPInputsCache([
            ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
            ['self_attn.o_proj'],
            ['mlp.up_proj', 'mlp.gate_proj'],
            ['mlp.down_proj']
        ])

    quantizers = {}
    observer = Observer()

    sequential = [
        ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        ['self_attn.o_proj'],
        ['mlp.up_proj', 'mlp.gate_proj'],
        ['mlp.down_proj']
    ]

    if requantize_layers is None:
        requantize_layers = set()

    for i in range(n_layers):
        # If pass2 and layer not selected: just forward to update activations
        if mode == "requantize" and i not in requantize_layers:
            layer = layers_s[i].to(dev)
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_embeddings=position_embeddings,
                                position_ids=position_ids)[0]
            layers_s[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            continue

        print(f'Quantizing layer {i+1}/{n_layers}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        # Otherwise: quantize / requantize this layer
        layer_s = layers_s[i].to(dev)
        full_s = find_layers(layer_s)
 
        # build teacher fp_inp cache for this layer (only for gptaq/greedyaq)
        if args.method in ["gptaq", "greedyaq"]:
            layer_t = layers_t[i].to(dev)
            full_t = find_layers(layer_t)

            fp_inputs_cache.add_hook(full_t)
            for j in range(args.nsamples):
                fp_inps[j] = layer_t(fp_inps[j].unsqueeze(0),
                                     attention_mask=attention_mask,
                                     position_embeddings=position_embeddings,
                                     position_ids=position_ids)[0]
            fp_inputs_cache.clear_hook()

        # collect Hessian stats on student
        layer_gptq = {}

        for names in sequential:
            subset_student = {n: full_s[n] for n in names}
            gptq = {}

            for name in subset_student:
                if args.method == "greedyaq":
                    gptq[name] = GreedyAQ(subset_student[name], observe=args.observe)
                elif args.method == "gptaq":
                    gptq[name] = GPTAQ(subset_student[name], observe=args.observe)
                elif args.method == "gptq":
                    gptq[name] = GPTQ(subset_student[name], observe=args.observe)
                elif args.method == "foem":
                    gptq[name] = FOEM(subset_student[name], observe=args.observe)
                else:
                    raise ValueError(f"Method {args.method} not supported.")

                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                if args.method in ["gptaq", "greedyaq"]:
                    gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            # share H/dXXT across grouped modules (like your existing code)
            first_module_name = list(subset_student.keys())[0]
            handle = subset_student[first_module_name].register_forward_hook(add_batch(first_module_name))

            for j in range(args.nsamples):
                outs[j] = layer_s(inps[j].unsqueeze(0),
                                  attention_mask=attention_mask,
                                  position_embeddings=position_embeddings,
                                  position_ids=position_ids)[0]

            handle.remove()

            if args.method in ["gptaq", "greedyaq"]:
                for name in subset_student:
                    if name != first_module_name:
                        gptq[name].H = gptq[first_module_name].H
                        gptq[name].dXXT = gptq[first_module_name].dXXT

            # quantize (pass1) or requantize with KD grad (pass2)
            for name in subset_student:
                kd_grad = None
                if mode == "requantize" and args.method == "greedyaq":
                    # fetch gradient for this layer/module
                    kd_grad = kd_gradients[i][name]

                    # IMPORTANT: reload FP teacher weights before re-quantization
                    # so we quantize the FP weights under KD-augmented objective
                    gptq[name].layer.weight.data = full_t[name].weight.data.clone().to(dev)

                if args.method == "greedyaq":
                    scale, zero, g_idx, error = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.groupsize,
                        actorder=args.act_order,
                        name=name,
                        alpha=args.alpha,
                        beta=args.beta,
                        gradient=kd_grad,
                        args=args,
                    )
                else:
                    scale, zero, g_idx, error = gptq[name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.groupsize,
                        actorder=args.act_order,
                        name=name,
                        alpha=args.alpha,
                        beta=args.beta,
                        args=args,
                    )

                quantizers[f"model.layers.{i}.{name}"] = (
                    gptq[name].quantizer.cpu(),
                    scale.cpu(), zero.cpu(), g_idx.cpu(),
                    args.wbits, args.groupsize
                )

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                if not args.observe:
                    gptq[name].free()

        # update student activations for next layer
        for j in range(args.nsamples):
            outs[j] = layer_s(inps[j].unsqueeze(0),
                              attention_mask=attention_mask,
                              position_embeddings=position_embeddings,
                              position_ids=position_ids)[0]

        if args.method in ["gptaq", "greedyaq"]:
            fp_inputs_cache.clear_cache()
            layers_t[i] = layer_t.cpu()
            del layer_t

        layers_s[i] = layer_s.cpu()
        del layer_s
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    if args.observe:
        observer.print()

    student_model.config.use_cache = use_cache
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


    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    bsz, seqlen = position_ids.shape
    cache_position = torch.arange(seqlen, device=dev)

    # rotary_emb only needs x for dtype/device; values don't matter
    dummy = torch.empty((1, seqlen, model.config.hidden_size), device=dev, dtype=inps.dtype)

    # (cos, sin) tuple
    position_embeddings = model.model.rotary_emb(dummy, position_ids)


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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, 
                            position_embeddings=position_embeddings, 
                            position_ids=position_ids)[0]
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
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.bfloat16)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.bfloat16)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.bfloat16)
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
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    # if eval:
    #     quant.make_quant_attn(model)
    #     quant.make_quant_norm(model)
    #     if fused_mlp:
    #         quant.make_fused_mlp(model)

    # if warmup_autotune:
    #     quant.autotune_warmup_linear(model, transpose=not (eval))
    #     if eval and fused_mlp:
    #         quant.autotune_warmup_fused(model)
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
    parser.add_argument('--alpha', type=float, default=0.25, help='Coefficient for weight correction term')
    parser.add_argument('--alpha-method', type=str, default="fixed", choices=["fixed", "sample", "optimize"], help='Coefficient for weight correction term')
    parser.add_argument('--mixup-param', type=float, default=5.0, help='Coefficient for weight correction term')
    parser.add_argument('--beam-size', type=int, default=1, help='Coefficient for weight correction term')
    parser.add_argument('--beam-cands', type=int, default=128, help='Coefficient for weight correction term')

    parser.add_argument('--beta', type=float, default=0.0003, help='Coefficient for weight correction term')
    parser.add_argument('--n_layers_to_update', type=int, default=5, help='Number of layers to update in second quantization.')
    parser.add_argument('--incoh-process', action='store_true', help='Whether to perform incoherence process.')
    parser.add_argument('--incoh-mode', type=str, default='kron', choices=['had', 'kron'], help='Incoherence mode for GreedyAQ.')
    parser.add_argument('--rescale-WH', action='store_true', help='Whether to rescale W and H to minimize proxy loss.')
    parser.add_argument('--ours', action='store_true', help='Use our method')
    parser.add_argument('--ours_v2', action='store_true', help='Use our method')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='llm-quantization', help='Wandb project name')
    parser.add_argument('--wandb-name', type=str, default='', help='Wandb run name (default: auto-generated)')
    parser.add_argument('--kd-T', type=float, default=1.0, help='Temperature for knowledge distillation')
    parser.add_argument('--kd-beta', type=float, default=1e-4, help='Beta for knowledge distillation')
    parser.add_argument('--first-quant-checkpoint', type=str, default='', help='Path to save/load first quantization checkpoint')
    parser.add_argument('--layers-to-update', type=str, default='sensitive', help='Comma-separated list of layer indices to update in second quantization (e.g., "0,1,2" or "all" or "sensitive" for sensitivity-based selection)')
    parser.add_argument('--sensitivity-threshold', type=float, default=None, help='Sensitivity threshold for selecting layers to update (only used if --layers-to-update=sensitive)')

    args = parser.parse_args()

    # Initialize wandb if enabled
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
                'beta': args.beta,
                'act_order': args.act_order,
                'true_sequential': args.true_sequential,
                'incoh_process': args.incoh_process,
                'incoh_mode': args.incoh_mode,
                'rescale_WH': args.rescale_WH,
                'n_layers_to_update': args.n_layers_to_update,
                'kd_T': args.kd_T,
                'kd_beta': args.kd_beta,
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
        teacher_model = get_llama(args.model)
        teacher_model.eval()

    dataloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=teacher_model.seqlen)

    quantizers = {}  # Initialize quantizers dict
    if (not args.load and args.wbits < 16 and not args.nearest) or args.step:
        # # Default to gptq if method not specified
        model = copy.deepcopy(teacher_model)
        model.eval()
        # build teacher + student
        teacher_model = get_llama(args.model)
        teacher_model.eval()

        student_model = copy.deepcopy(teacher_model)
        student_model.eval()

        dataloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed,
                                model=args.model, seqlen=teacher_model.seqlen)

        # -------- PASS 1: regular GreedyAQ quantization (no KD) --------
        tick = time.time()
        quantizers = llama_sequential(
            student_model,
            dataloader,
            DEV,
            teacher_model=teacher_model,   # needed for fp_inp cache (GreedyAQ/GPTAQ)
            args=args,
            kd_gradients=None,
            requantize_layers=None,
            mode="quantize",
        )
        t1 = time.time() - tick
        print(f"First quantization time: {t1:.2f}s")
        if args.wandb: wandb.log({"first_quant_time": t1})

        # -------- Compute KD gradients ONCE: teacher vs quantized student --------
        use_layerwise_kd = (args.method == "greedyaq" and getattr(args, "kd_beta", 0.0) != 0.0)
        if use_layerwise_kd:
            with torch.enable_grad():
                kd_gradients, _ = gradient_utils.get_kd_gradients(
                    teacher_model=teacher_model,
                    student_model=student_model,
                    dataloader=dataloader,
                    num_batches=args.nsamples,
                    T=getattr(args, "kd_T", 1.0),
                    dev=DEV,
                    save_path=None,
                )

            # choose layers to requantize
            requant_layers = parse_layers_to_update(args, len(student_model.model.layers))
            print("Requantize layers:", sorted(list(requant_layers))[:10], "...")

            # -------- PASS 2: requantize selected layers using KD gradients --------
            tick = time.time()
            quantizers = llama_sequential(
                student_model,
                dataloader,
                DEV,
                teacher_model=teacher_model,
                args=args,
                kd_gradients=kd_gradients,
                requantize_layers=requant_layers,
                mode="requantize",
            )
            t2 = time.time() - tick
            print(f"Second (KD) requantization time: {t2:.2f}s")
            if args.wandb: wandb.log({"second_quant_time": t2})
        model = student_model 
        
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

    if not args.observe and args.save_safetensors:
        llama_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

    if args.wandb:
        wandb.finish()