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
        torch.manual_seed(self.seed)

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

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0
        if G is not None:
            G[:, dead] = 0
            
        D = self.dXXT.clone()

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
        

        p = torch.argsort(torch.diag(Hr), descending=False) # sort based on column of X_t 
        inv_p = torch.argsort(p)
        Hp = Hr[p][:, p]
        Mp = Mr[p][:, p]
        
        L = torch.linalg.cholesky(Hp)
        Hp_inv = torch.cholesky_inverse(L)

        L_diag = torch.diag(L)
        L = L / L_diag.unsqueeze(0)  # Broadcast division: each column divided by its diagonal
        L = L - torch.eye(L.shape[0], device=L.device)

        C = Mp @ Hp_inv
        W_ref = Wr[:, p] @ C 
        del C, Mr

        if not self.quantizer.ready():
            self.quantizer.find_params(W_ref, weight=True)
        
        # ===============================
        # Global (across-all-columns) blockwise beam rounding
        # Keeps beams across blocks; pick best ONLY at the very end.
        # ===============================

        rows, cols = W_ref.shape
        device = W_ref.device

        beam_width = int(getattr(args, "beam_size", 1))
        beam_width = max(1, beam_width)
        B = beam_width

        beam_cands = int(getattr(args, "beam_cands", 0))
        beam_cands = max(0, beam_cands)

        maxq = int(self.quantizer.maxq.item())
        A = maxq + 1
        codes = torch.arange(A, device=device)   # for arithmetic

        # Use float32 for stable scoring
        W_ref_f = W_ref.float()
        L_f = L.float()
        L_diag_f = L_diag.float()

        # Beam scores: start with 1 active beam (beam 0), rest inactive (inf)
        beam_scores = torch.full((rows, B), float("inf"), device=device)
        beam_scores[:, 0] = 0.0

        # We store the already-quantized suffix for each beam as we move left.
        # Q_tail always corresponds to columns [i2:cols] at the start of each block.
        tail_dtype = torch.float32
        Q_tail = torch.empty((rows, B, 0), device=device, dtype=tail_dtype)  # [rows,B,tail_len]

        # For returning packing params
        scale = []
        zero = []
        curr_gid = None
        seen_groups = set() 
        
        # NOTE: blocksize can be your input blocksize (e.g., 128). Don't overwrite it to cols.
        for i2 in range(cols, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1
            tail_len = Q_tail.shape[2]   # should equal cols - i2

            # Block data
            W1 = W_ref_f[:, i1:i2]                    # [rows,count]
            Lblk = L_f[i1:i2, i1:i2]                  # [count,count]

            # Beam-dependent tail correction:
            # tail_corr[b] = (W_ref_tail - Q_tail[b]) @ Ltail
            if tail_len > 0:
                # W_ref tail for this block is columns [i2:cols]
                W2 = W_ref_f[:, None, i2:]            # [rows,1,tail_len]
                E2 = W2 - Q_tail             # [rows,B,tail_len]
                Ltail = L_f[i2:, i1:i2]               # [tail_len,count]
                tail_corr = E2 @ Ltail   # [rows,B,count]
            else:
                tail_corr = torch.zeros((rows, B, count), device=device)

            # Beam state for THIS block: quantized values for columns [i1:i2]
            beam_states = torch.zeros((rows, B, count), device=device, dtype=torch.float32)

            # Track which old-tail beam each current beam inherits from (so we can reorder Q_tail once per block)
            tail_src = torch.arange(B, device=device, dtype=torch.long).view(1, B).expand(rows, B)  # [rows,B]

            # Decode columns in this block from right to left
            for ii in reversed(range(count)):
                col_abs = i1 + ii
                Ljj2 = (L_diag_f[col_abs] ** 2)   # scalar tensor

                if groupsize != -1:
                    gstart = (i1 + ii) // groupsize * groupsize
                    gend   = min(gstart + groupsize, self.columns)
                    group_id = (i1 + ii) // groupsize

                    if group_id not in seen_groups:
                        self.quantizer.find_params(W_ref[:, gstart:gend], weight=True)
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        seen_groups.add(group_id)

                sc = self.quantizer.scale.reshape(rows, 1).float()  # [rows,1]
                ze = self.quantizer.zero.reshape(rows, 1).float()   # [rows,1]

                # What = W1[:,ii] + (W1@v - Qblock@v) + tail_corr[:, :, ii]
                v = Lblk[:, ii]                       # [count]
                What = W1[:, ii].unsqueeze(1) + (W1.unsqueeze(1) - beam_states) @ v + tail_corr[:, :, ii]

                # if beam_width == 1:
                #     q = self.quantizer.quantize(What[:, 0].unsqueeze(1)).flatten()
                #     beam_states[:, 0, ii] = q
            
                #     u = What / sc + ze  
                #     du = u.unsqueeze(-1) - codes.view(1,1,A) 
                #     inc = (du * sc.unsqueeze(1)).pow(2) * Ljj2 
                #     beam_scores = inc 
                    # continue

                # full alphabet
                levels = (codes.view(1, A) - ze) * sc          # [rows,A]
                u = What / sc + ze                             # [rows,B]
                du = u.unsqueeze(-1) - codes.view(1,1,A)         # [rows,B,A]
                inc = (du * sc.unsqueeze(1)).pow(2) * Ljj2        # [rows,B,A]
                # inc = (What.unsqueeze(-1) - levels.unsqueeze(1)).pow(2) * Ljj2  # [rows,B,A]
                
                new_scores = beam_scores.unsqueeze(-1) + inc  # [rows,B,A]

                flat = new_scores.reshape(rows, -1)   
                keep = B
                topv, topi = torch.topk(flat, k=keep, dim=1, largest=False, sorted=True)

                parent = topi // A
                choice = topi % A

                gather_idx = parent.unsqueeze(-1).expand(-1, -1, count)
                beam_states = beam_states.gather(1, gather_idx).clone()
                tail_corr = tail_corr.gather(1, parent.unsqueeze(-1).expand(-1, -1, count)).clone()
                tail_src = tail_src.gather(1, parent)
                beam_scores = topv

                q_sel = levels.gather(1, choice)    # [rows,B]
                beam_states[:, :, ii] = q_sel


            # ---- End of block: build the new suffix Q_tail = [this block | previous tail] per beam ----
            if tail_len > 0:
                # reorder old tail to match final beam ancestry
                idx = tail_src.unsqueeze(-1).expand(-1, -1, tail_len)     # [rows,B,tail_len]
                old_tail = Q_tail.gather(1, idx)                          # [rows,B,tail_len]
                Q_tail = torch.cat([beam_states.to(tail_dtype), old_tail], dim=2)  # [rows,B,count+tail_len]
            else:
                Q_tail = beam_states.to(tail_dtype)                       # [rows,B,count]

        # After all blocks, Q_tail is [rows,B,cols] (in permuted column order)
        best = beam_scores.argmin(dim=1)                                  # [rows]
        Qp = Q_tail[torch.arange(rows, device=device), best, :].to(W_ref.dtype)  # [rows,cols]
        obj_total = float(beam_scores.min(dim=1).values.sum().item())

        # Undo permutation
        Q = Qp[:, inv_p]


        # # ===============================
        # # Lazy blockwise beam rounding (per-row top-k), no global corr/q_states
        # # ===============================
        # rows, cols = W_ref.shape
        # device = W_ref.device

        # beam_width = int(getattr(args, "beam_size", 1))
        # beam_width = max(1, beam_width)

        # # Optional: restrict candidates around the nearest grid point
        # # 0 or >=A => use full alphabet
        # beam_cands = int(getattr(args, "beam_cands", 0))
        # beam_cands = max(0, beam_cands)

        # maxq = int(self.quantizer.maxq.item())
        # A = maxq + 1
        # codes = torch.arange(A, device=device)  # [A]

        # # Output in permuted space (same as W_ref)
        # Qp = torch.zeros_like(W_ref)

        # # Track proxy objective (sum over rows) for your logging
        # obj_rows = torch.zeros(rows, device=device)

        # g_idx = []
        # scale = []
        # zero = []
        # seen_groups = set() 
        # curr_gid = None 

        # for i2 in range(cols, 0, -blocksize):
        #     i1 = max(i2 - blocksize, 0)
        #     count = i2 - i1

        #     # Current block data
        #     W1 = W_ref[:, i1:i2]                 # [rows, count]
        #     Lblk = L[i1:i2, i1:i2]               # [count, count]  (strictly lower, diag=0)

        #     # Precompute W1 @ Lblk once (so W1@v is just a column slice)
        #     W1_Lblk = W1 @ Lblk                    # [rows, count]

        #     # Tail correction from already-fixed suffix
        #     if i2 < cols:
        #         E2 = (W_ref[:, i2:] - Qp[:, i2:])          # [rows, cols-i2]
        #         Ltail = L[i2:, i1:i2]                               # [cols-i2, count]
        #         tail_corr = E2 @ Ltail                                # [rows, count]
        #     else:
        #         tail_corr = torch.zeros((rows, count), device=device)

        #     # Beam state ONLY for this block: [rows, Bcur, count]
        #     # Start from current Q in this block (usually zeros at first encounter)
        #     beam_states = Qp[:, i1:i2].float().unsqueeze(1)           # [rows, 1, count]
        #     beam_scores = torch.zeros((rows, 1), device=device)

        #     for ii in range(count - 1, -1, -1):
        #         if groupsize != -1:
        #             gstart = (i1 + ii) // groupsize * groupsize
        #             gend   = min(gstart + groupsize, self.columns)
        #             group_id = (i1 + ii) // groupsize

        #             if group_id not in seen_groups:
        #                 self.quantizer.find_params(W_ref[:, gstart:gend], weight=True)
        #                 scale.append(self.quantizer.scale)
        #                 zero.append(self.quantizer.zero)
        #                 seen_groups.add(group_id)
                
        #         Ljj2 = (L_diag[i1 + ii] ** 2)        # scalar

        #         # v = Lblk[:, ii] contains L[k, col_abs] for k in this block
        #         v = Lblk[:, ii]                                       # [count]

        #         # What = W1[:,ii] + (W1 - Qblock) @ v + tail_corr
        #         W1v = W1_Lblk[:, ii]                                  # [rows]
        #         Qv  = torch.matmul(beam_states, v)                    # [rows, Bcur]
        #         What = W1[:, ii].unsqueeze(1) + (W1v.unsqueeze(1) - Qv) + tail_corr[:, ii].unsqueeze(1)  # [rows, Bcur]
               
        #         if beam_width <= 1:
        #             q_greedy = self.quantizer.quantize(What.squeeze(1).unsqueeze(1)).flatten()
        #             beam_states[:, 0, ii] = q_greedy
        #             beam_scores[:, 0] += (What[:, 0] - q_greedy).pow(2) * Ljj2
        #             continue

        #         # Full alphabet candidates (exact over grid)
        #         sc = self.quantizer.scale.float()
        #         ze = self.quantizer.zero.float()
        #         levels = (codes.view(1, -1) - ze.reshape(rows, 1)) * sc.reshape(rows, 1)              # [rows, A]
        #         u = What / sc + ze
                
        #         # full alphabet code grid
        #         du = u.unsqueeze(-1) - codes.view(1,1,A)                       # [rows,Bcur,A]
        #         inc = (du * sc.unsqueeze(1)).pow(2) * Ljj2

        #         new_scores = beam_scores.unsqueeze(-1) + inc                               # [rows,Bcur,A]
        #         flat = new_scores.reshape(rows, -1)                                        # [rows,Bcur*A]

        #         keep = min(beam_width, flat.shape[1])
        #         topv, topi = torch.topk(flat, k=keep, dim=1, largest=False, sorted=True)  # [rows,keep]

        #         parent = topi // A
        #         choice = topi % A

        #         gather_idx = parent.unsqueeze(-1).expand(-1, -1, count)
        #         beam_states = beam_states.gather(1, gather_idx).clone()
        #         beam_scores = topv
                
        #         q_sel = levels.gather(1, choice)                                           # [rows,keep]
        #         beam_states[:, :, ii] = q_sel

        #     # Commit the best beam PER ROW for this block
        #     best = beam_scores.argmin(dim=1)                                                   # [rows]
        #     Qp[:, i1:i2] = beam_states[torch.arange(rows, device=device), best, :].to(W_ref.dtype)
        #     obj_rows += beam_scores.min(dim=1).values

        # # Final proxy objective (sum over rows)
        # obj_total = float(obj_rows.sum().item())

        # # Undo column permutation back to original order
        # Q = Qp[:, inv_p]

        if args.incoh_process:
            Q = incoherence_process(Q, SU, SV, scaleWH, args)

        torch.cuda.synchronize()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        g_idx = g_idx[inv_p] 

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # todo
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        self.print_loss(name=name, q_weight=Q, alpha=obj_total, timecost=(time.time() - tick))
        
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        
        # Reverse scale and zero to match original column order since we processed in reverse
        scale = torch.cat(scale[::-1], dim=1)
        zero = torch.cat(zero[::-1], dim=1)

        return scale, zero, g_idx, obj_total


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

