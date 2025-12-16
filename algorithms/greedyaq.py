import math
import os
import time
import logging
import torch
import torch.nn as nn
import transformers
import quant
from texttable import Texttable
from utils import torch_snr_error
import utils.quip_utils as quip_utils

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class Observer:

    def __init__(self, topk=32):
        self.loss_list = []
        self.topk = topk

    def submit(self, name: str, layerid: int, gptq, error: float):

        item = (name, layerid, {'gptq': gptq, 'error': error})

        if len(self.loss_list) < self.topk:
            self.loss_list.append(item)
            return

        min_error = error
        min_idx = -1
        for idx, data in enumerate(self.loss_list):
            if min_error > data[2]['error']:
                min_idx = idx
                min_error = data[2]['error']

        if min_idx >= 0:
            self.loss_list[min_idx] = item

    def print(self):
        self.loss_list = sorted(self.loss_list, key=lambda s: s[2]['error'], reverse=True)

        table = Texttable()

        table.header(['name', 'error'])
        table.set_cols_dtype(['t', 'f'])

        for item in self.loss_list:
            table.add_row([f"{item[0]}.{item[1]}", item[2]['error']])
        print(table.draw())
        print('\n')

    def items(self):
        return self.loss_list


class GreedyAQ:

    def __init__(self, layer, observe=False, store_delta_x=False, sampled_alpha=False, mixup_param=0.5, seed=42):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        # self.dXdXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.inp1 = None
        self.out1 = None
        self.nsamples = 0
        self.quantizer = quant.Quantizer()
        self.observe = observe
        self.inps = []
        self.store_delta_x = store_delta_x
        self.delta_x_values = [] if store_delta_x else None  # Store |deltaX| values for plotting

        # sampling 
        self.sampled_alpha = sampled_alpha
        self.mixup_param = mixup_param
        self.seed = seed

    def add_batch(self, inp, out):
        if self.observe:
            self.inp1 = inp
            self.out1 = out
        else:
            self.inp1 = None
            self.out1 = None


        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        # self.dXdXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
        # I'll sample alpha here - from Beta distribution (to use different sampled alpha for different calibration sample)
        if self.sampled_alpha:
            torch.manual_seed(self.seed)
            self._beta_dist = torch.distributions.Beta(self.mixup_param, self.mixup_param)
            alpha = self._beta_dist.sample().item()
            alpha = min(alpha, 1-alpha)
            dX = dX * alpha
        self.dXXT += dX.matmul(inp.t())
        # self.dXdXT += dX.matmul(dX.t())
        
        # Store |deltaX| for plotting only if enabled: shape is [channels, samples]
        if self.store_delta_x:
            abs_dX = torch.abs(dX)  # |deltaX| per channel
            self.delta_x_values.append(abs_dX.cpu().clone())
        
        del self.fp_inp[0]


    def print_loss(self, name, q_weight, alpha, timecost):
        table = Texttable()
        name += ' ' * (16 - len(name))

        table.header(['name', 'alpha', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.inp1 is not None:
            # quantize input to int8
            quantizer = quant.Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1, weight=True)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'

        table.add_row([name, alpha, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])


    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name='', fp_weight=None, alpha=0.25, beta=None, gradient=None, args=None):
        self.layer.to(self.dev)

        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        H = self.H

        G = None
        if gradient is not None:
            G = gradient.to(self.dev).float()
            if isinstance(self.layer, nn.Conv2d):
                G = G.flatten(1)

        beta = getattr(args, "kd_beta", 1e-4) 

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0
        if G is not None:
            G[:, dead] = 0

        D = self.dXXT.clone()
        del self.dXXT

        if args.incoh_process:
            Hr, Dr, Wr, SU, SV, scaleWH = incoherence_preprocess(W, H, D, args)
        else:
            Hr = H
            Dr = D
            Wr = W
            SU = None
            SV = None
            scaleWH = None

        damp = args.percdamp * torch.mean(torch.diag(Hr))
        diag = torch.arange(Hr.shape[0], device=Hr.device)
        Hr[diag, diag] += damp

        if self.sampled_alpha is not None and self.sampled_alpha:
            Mr = Hr + Dr
        else:
            Mr = Hr + args.alpha * Dr
        
        if not self.quantizer.ready():
            self.quantizer.find_params(Wr, weight=True)
        
        p = torch.argsort(torch.diag(Hr), descending=False) # sort based on column of X_t 
        inv_p = torch.argsort(p)
        # P = torch.eye(Hr.shape[0], device=Hr.device)[:, p]
        Hp = Hr[p][:, p]
        Mp = Mr[p][:, p]

        L = torch.linalg.cholesky(Hp)
        Hp_inv = torch.cholesky_inverse(L)

        Delta_W = None
        if G is not None and beta != 0.0:
            Delta_W = (0.5 * beta * G[:, p]) @ Hp_inv   # [rows, cols]
        
        L_diag = torch.diag(L)
        L = L / L_diag.unsqueeze(0)  # Broadcast division: each column divided by its diagonal
        L = L - torch.eye(L.shape[0], device=L.device)
        del L_diag

        C = Mp @ Hp_inv
        W_ref = Wr[:, p] @ C 
        del C, Mr

        if Delta_W is not None:
            W_ref = W_ref - Delta_W
            del Delta_W

        if getattr(args, "sort_asym", False):
            H_diag = torch.diag(Hr)
            W_norm2 = (W_ref ** 2).sum(dim=0)
            scores = H_diag * W_norm2          # GPTQ-like
            p_asym = torch.argsort(scores, descending=True)  # or False if you prefer
            Hp = Hr[p_asym][:, p_asym]
            W_ref = W_ref[:, p_asym]

            L = torch.linalg.cholesky(Hp)
            L_diag = torch.diag(L)
            L = L / L_diag.unsqueeze(0)  # Broadcast division: each column divided by its diagonal
            L = L - torch.eye(L.shape[0], device=L.device)
            del L_diag

        Q = torch.zeros_like(W_ref)

        g_idx = []
        scale = []
        zero = []
        seen_groups = set()  # Track which groups we've already saved
        
        for i2 in range(self.columns, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1
            W1 = W_ref[:, i1:i2].clone()
            W2diff = W_ref[:, i2:] - Q[:, i2:]
            What1 = Q[:, i1:i2].clone()
            L1 = L[:, i1:i2]
            
            for i in reversed(range(count)):
                if groupsize != -1:
                    gstart = (i1 + i) // groupsize * groupsize
                    gend   = min(gstart + groupsize, self.columns)
                    group_id = (i1 + i) // groupsize

                    if group_id not in seen_groups:
                        self.quantizer.find_params(W_ref[:, gstart:gend], weight=True)
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        seen_groups.add(group_id)

                What = W1[:,i] + (W1 - What1) @ L1[i1:i2,i] + W2diff @ L1[i2:,i]
                What1[:, i] = self.quantizer.quantize(What.unsqueeze(1)).flatten()
            Q[:, i1:i2] = What1


        if getattr(args, "sort_asym", False):
            Q = Q[:, torch.argsort(p_asym)] # inverse perm

        Q = Q[:, inv_p].to(Q.device)

        if args.alpha_method == "fixed":
            first = ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj']
            if len(args.alpha_per_module[name]) == 1: 
                alpha = args.alpha
            else:
                diff = Q - Wr
                WD = Wr @ Dr
                num = torch.trace(diff.t() @ WD).float()
                WDP = WD[:, p]
                denom = torch.trace((WDP @ Hp_inv) @ WDP.t()).float()
                alpha = torch.clamp(num / denom, 0.0, 1.0).item()    
                del WD, WDP, diff, Wr, Dr, Hp_inv
            args.alpha_per_module[name].append(alpha)
            args.alpha_track.append(alpha)


        if args.incoh_process:
            Q = incoherence_process(Q, SU, SV, scaleWH, args)
        torch.cuda.synchronize()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # todo
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        self.print_loss(name=name, q_weight=Q, alpha=args.alpha, timecost=(time.time() - tick))
        
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        
        # Reverse scale and zero to match original column order since we processed in reverse
        scale = torch.cat(scale[::-1], dim=1)
        zero = torch.cat(zero[::-1], dim=1)

        return scale, zero, g_idx, None


    def fasterquant_alternating_method(
            self, gradient=None, blocksize=128, percdamp=.01, groupsize=-1, name='', actorder=False,  alpha=0.25, args=None,
    ):


        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H

        G = None
        if gradient is not None:
            G = gradient.to(self.dev).float()
            if isinstance(self.layer, nn.Conv2d):
                G = G.flatten(1)

        beta = getattr(args, "kd_beta", 1e-4) 
        D = self.dXXT.clone()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0
        # self.dXdXT[:, dead] = 0
        if G is not None:
            G[:, dead] = 0

        if getattr(args, "rescale_D", False):
            s = D.norm(dim=1)              # [d]
            s = torch.clamp(s, min=1e-8)   
            print(s) 
            D = D / s[:, None]

        if args.incoh_process:
            Hr, Dr, Wr, SU, SV, scaleWH = incoherence_preprocess(W, H, D, args)
        else:
            Hr = H
            Dr = D
            Wr = W
            SU = None
            SV = None
            scaleWH = None

        damp = percdamp * torch.mean(torch.diag(Hr))
        diag = torch.arange(self.columns, device=self.dev)
        Hr[diag, diag] += damp 

        p = torch.argsort(torch.diag(Hr), descending=False) # sort based on column of X_t 
        inv_p = torch.argsort(p)
        # P = torch.eye(Hr.shape[0], device=H.device)[:, p]
        Hp = Hr[p][:, p]
        L = torch.linalg.cholesky(Hp)
  
        Hp_inv = torch.cholesky_inverse(L)
        
        Delta_W = None
        if G is not None and beta != 0.0:
            Delta_W = (0.5 * beta * G[:, p]) @ Hp_inv   # [rows, cols]

        L_diag = torch.diagonal(L)                    # view, no big alloc
        L.div_(L_diag.unsqueeze(0))                   # in-place (no new [d,d])
        L.fill_diagonal_(0.0)                         # in-place (no eye)
        del L_diag 
            
        # L_diag = torch.diag(L)
        # L = L / L_diag.unsqueeze(0)  # Broadcast division: each column divided by its diagonal
        # L = L - torch.eye(L.shape[0], device=L.device)
        # del L_diag

        W_ref = Wr.clone()
        Q = torch.zeros_like(W_ref)

        alpha_update = args.alpha
        num_iterations = 2
        best_error = float('inf')

        for iter in range(num_iterations):
            
            Mr = Hr + alpha_update * Dr
            C = Mr[:, p] @ Hp_inv
            W_new = W_ref @ C 
            if Delta_W is not None:
                W_new = W_new - Delta_W

            Q = torch.zeros_like(W_ref)

            g_idx = []
            scale = []
            zero = []
            seen_groups = set()  # Track which groups we've already saved
            
            for i2 in range(self.columns, 0, -blocksize):
                i1 = max(i2 - blocksize, 0)
                count = i2 - i1
                W1 = W_new[:, i1:i2].clone()
                W2diff = W_new[:, i2:] - Q[:, i2:]
                What1 = Q[:, i1:i2].clone()
                L1 = L[:, i1:i2]
                
                for i in reversed(range(count)):
                    if groupsize != -1:
                        gstart = (i1 + i) // groupsize * groupsize
                        gend   = min(gstart + groupsize, self.columns)
                        group_id = (i1 + i) // groupsize

                        if group_id not in seen_groups:
                            self.quantizer.find_params(W_new[:, gstart:gend], weight=True)
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            seen_groups.add(group_id)

                    What = W1[:,i] + (W1 - What1) @ L1[i1:i2,i] + W2diff @ L1[i2:,i]
                    What1[:, i] = self.quantizer.quantize(What.unsqueeze(1)).flatten()
                Q[:, i1:i2] = What1
            
            error = torch.trace((W_ref - Q[:, inv_p]) @ H @ ((W_ref - Q[:, inv_p]).t()))
            error_val = error.item()
            
            if error_val >= best_error:
                break
            else:
                best_error = error_val
                diff = Q[:, inv_p] - W_ref
                WD = W_ref @ D
                num = torch.trace(diff.t() @ WD)
                WD_P = WD[:, p]
                denom = torch.trace(WD_P @ Hp_inv @ WD_P.t())
                if denom != 0:
                    alpha_update = torch.clamp(num / denom, 0, 1.0)
                    print(f"Iteration {iter}: error {error_val:.6f}, alpha {alpha_update:.3f}")
                else:
                    break
        
        if ("down" in name or "o_proj" in name): # last layer of a block then update alpha 
            name = name + f"_a{alpha_update:.2f}"
            if "o_proj" in name: 
                args.alpha_attn = alpha_update
            if "down" in name:
                args.alpha_mlp = alpha_update  
            
            args.alpha_track.append(alpha_update)

        Q = Q[:, inv_p].to(Q.device)
        if args.incoh_process:
            Q = incoherence_process(Q, SU, SV, scaleWH, args)

        torch.cuda.synchronize()
        
        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        # todo
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        self.print_loss(name=name, q_weight=Q, alpha=alpha_update, timecost=(time.time() - tick))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        # Reverse scale and zero to match original column order since we processed in reverse
        scale = torch.cat(scale[::-1], dim=1)
        zero = torch.cat(zero[::-1], dim=1)
        g_idx = g_idx[inv_p]

        return scale, zero, g_idx, None
    

    # def fasterquant_alternating_method(
    #         self, gradient=None, blocksize=128, percdamp=.01, groupsize=-1, name='', actorder=False,  alpha=0.25, args=None,
    # ):


    #     W = self.layer.weight.data.clone()
    #     W = W.float()

    #     tick = time.time()

    #     if not self.quantizer.ready():
    #         self.quantizer.find_params(W, weight=True)

    #     H = self.H
    #     del self.H

    #     G = None
    #     if gradient is not None:
    #         G = gradient.to(self.dev).float()
    #         if isinstance(self.layer, nn.Conv2d):
    #             G = G.flatten(1)

    #     beta = getattr(args, "kd_beta", 1e-4) 

    #     dead = torch.diag(H) == 0
    #     H[dead, dead] = 1
    #     W[:, dead] = 0
    #     self.dXXT[:, dead] = 0
    #     if G is not None:
    #         G[:, dead] = 0

    #     damp = percdamp * torch.mean(torch.diag(H))
    #     diag = torch.arange(self.columns, device=self.dev)
    #     H[diag, diag] += damp

    #     p = torch.argsort(torch.diag(H), descending=False) # sort based on column of X_t 
    #     P = torch.eye(H.shape[0], device=H.device)[:, p]
    #     Hp = H[p][:, p]
    #     L = torch.linalg.cholesky(Hp)

    #     D = self.dXXT.clone()

    #     if torch.norm(D, 'fro') > 0:
    #         norm_ratio = torch.norm(D, 'fro') / torch.norm(H, 'fro')
    #         alpha = norm_ratio
    #     else:
    #         # If D is zero, use the provided alpha value
    #         alpha = 0
  
    #     Hp_inv = torch.cholesky_inverse(L)
        
    #     Delta_W = None
    #     if G is not None and beta != 0.0:
    #         Delta_W = (0.5 * beta * G[:, p]) @ Hp_inv   # [rows, cols]
        
    #     L_diag = torch.diag(L)
    #     L = L / L_diag.unsqueeze(0)  # Broadcast division: each column divided by its diagonal
    #     L = L - torch.eye(L.shape[0], device=L.device)
    #     del L_diag

    #     W_ref = W.clone()
    #     Q = torch.zeros_like(W_ref)

    #     alpha_update = 0.25
    #     num_iterations = 3
    #     best_error = float('inf')

    #     for iter in range(num_iterations):
            
    #         M = H + alpha_update * D
    #         C = M[:, p] @ Hp_inv
    #         W_new = W_ref @ C 
    #         if Delta_W is not None:
    #             W_new = W_new - Delta_W

    #         Q = torch.zeros_like(W_ref)

    #         g_idx = []
    #         scale = []
    #         zero = []
    #         seen_groups = set()  # Track which groups we've already saved
            
    #         for i2 in range(self.columns, 0, -blocksize):
    #             i1 = max(i2 - blocksize, 0)
    #             count = i2 - i1
    #             W1 = W_new[:, i1:i2].clone()
    #             W2diff = W_new[:, i2:] - Q[:, i2:]
    #             What1 = Q[:, i1:i2].clone()
    #             L1 = L[:, i1:i2]
                
    #             for i in reversed(range(count)):
    #                 if groupsize != -1:
    #                     gstart = (i1 + i) // groupsize * groupsize
    #                     gend   = min(gstart + groupsize, self.columns)
    #                     group_id = (i1 + i) // groupsize

    #                     if group_id not in seen_groups:
    #                         self.quantizer.find_params(W_new[:, gstart:gend], weight=True)
    #                         scale.append(self.quantizer.scale)
    #                         zero.append(self.quantizer.zero)
    #                         seen_groups.add(group_id)

    #                 What = W1[:,i] + (W1 - What1) @ L1[i1:i2,i] + W2diff @ L1[i2:,i]
    #                 What1[:, i] = self.quantizer.quantize(What.unsqueeze(1)).flatten()
    #             Q[:, i1:i2] = What1
            
    #         error = torch.trace((W_new - Q) @ Hp @ (W_new - Q).t())
    #         # error = torch.trace((W_ref - Q @ P.t()) @ H @ ((W_ref - Q @ P.t()).t()))
    #         error_val = error.item()
            
    #         if error_val >= best_error:
    #             break
    #         else:
    #             best_error = error_val
    #             diff = Q @ P.t() - W_ref
    #             num = torch.trace(diff.t() @ W_ref @ D)
    #             denom = torch.trace(W_ref @ self.dXdXT @ W_ref.t())
    #             if denom != 0:
    #                 alpha_update = torch.clamp(num / denom, 0.05, args.alpha)
    #                 print(f"Iteration {iter}: error {error_val:.6f}, alpha {alpha_update:.3f}")
    #             else:
    #                 break

    #     Q = Q @ P.t().to(Q.device)
        
    #     torch.cuda.synchronize()
        
    #     groupsize = groupsize if groupsize != -1 else self.columns
    #     g_idx = [i // groupsize for i in range(self.columns)]
    #     g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        

    #     if isinstance(self.layer, transformers.Conv1D):
    #         Q = Q.t()
        
    #     # todo
    #     self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
    #         self.layer.weight.data.dtype
    #     )

    #     self.print_loss(name=name, q_weight=Q, alpha=alpha_update, timecost=(time.time() - tick))

    #     if scale == []:
    #         scale.append(self.quantizer.scale)
    #         zero.append(self.quantizer.zero)
    #     # Reverse scale and zero to match original column order since we processed in reverse
    #     scale = torch.cat(scale[::-1], dim=1)
    #     zero = torch.cat(zero[::-1], dim=1)
    #     return scale, zero, g_idx, None
 

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.dXXT = None 
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


def RHT_H(H, SU):
    return quip_utils.matmul_hadUt(quip_utils.matmul_hadUt(H * SU).T * SU)


def RHT_W(W, SU, SV):
    return quip_utils.matmul_hadUt(quip_utils.matmul_hadUt(W.T * SV).T * SU)


def incoherence_preprocess(W, H, D, args):
    dtype_ = torch.float32
    device = H.device
    (m, n) = W.shape

    # diagonally rescale W,H to minimize proxy loss
    scaleWH = None
    Wr = W
    Hr = H
    Dr = D 
    if args.rescale_WH:
        Hr = H / H.abs().max()
        diagH = torch.diag(Hr)
        diagW2 = torch.diag(W.T @ W)
        diagH = torch.clamp(diagH, min=1e-8)
        diagW2 = torch.clamp(diagW2, min=1e-8)
        scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
        scaleWH = scaleWH.clamp(min=1e-8)
        Wr = Wr * scaleWH[None, :]
        Hr = Hr / scaleWH[None, :]
        Hr = Hr / scaleWH[:, None]
        if D is not None:
            Dr = Dr / scaleWH[None, :]
            Dr = Dr / scaleWH[:, None]
        scaleWH = scaleWH.cpu()

    # randomized hadamard transformation on H, W
    if args.incoh_mode == "had":
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
        Hr = RHT_H(Hr, SU)
        if D is not None:
            Dr = RHT_H(Dr, SU).T # transpose since D is not symmetric 
        Wr = RHT_W(Wr, SU, SV)
    
    # randomized kronecker product on H, W
    elif args.incoh_mode == "kron":
        SU = quip_utils.rand_ortho_butterfly_noblock(n).to(dtype_).to(device)
        SV = quip_utils.rand_ortho_butterfly_noblock(m).to(dtype_).to(device)
        Hr = SU @ Hr @ SU.T
        if D is not None:
            Dr = SU @ Dr @ SU.T
        Wr = SV @ Wr @ SU.T
    else:
        raise NotImplementedError
    SV = SV.cpu()
    SU = SU.cpu()

    # Handle dead columns after transformation
    dead = torch.diag(Hr) == 0
    Hr[dead, dead] = 1
    Wr[:, dead] = 0

    Wr = Wr.to(device)

    return Hr, Dr, Wr, SU, SV, scaleWH


def incoherence_process(hatWr, SU, SV, scaleWH, args):
    device = hatWr.device
    # reverse hadamard transformation
    if args.incoh_mode == 'had':
        hatWr = (quip_utils.matmul_hadU(
            (quip_utils.matmul_hadU(hatWr) * SU.to(device)).T) * SV.to(device)).T
    # reverse kronecker product
    elif args.incoh_mode == 'kron':
        hatWr = SV.T.to(device) @ hatWr @ SU.to(device)
    else:
        raise NotImplementedError

    # reverse rescale W,H
    if args.rescale_WH:
        hatWr /= scaleWH[None, :].to(device)

    assert torch.isfinite(hatWr).all()
    return hatWr

