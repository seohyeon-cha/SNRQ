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


class LDLQ:

    def __init__(self, layer, observe=False, store_delta_x=False):
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
        # self.dXdXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.inp1 = None
        self.out1 = None
        self.nsamples = 0
        self.quantizer = quant.Quantizer()
        self.observe = observe
        self.inps = []

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
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


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

        if G is not None:
            G[:, dead] = 0


        if args.incoh_process:
            Hr, _, Wr, SU, SV, scaleWH = incoherence_preprocess(W, H, None, args)
        else:
            Hr = H
            Wr = W
            SU = None
            SV = None
            scaleWH = None

        damp = args.percdamp * torch.mean(torch.diag(Hr))
        diag = torch.arange(Hr.shape[0], device=Hr.device)
        Hr[diag, diag] += damp

        
        if not self.quantizer.ready():
            self.quantizer.find_params(Wr, weight=True)

    
        p = torch.argsort(torch.diag(Hr), descending=False) # sort based on column of X_t 
        P = torch.eye(Hr.shape[0], device=Hr.device)[:, p]
        Hp = Hr[p][:, p]

        L = torch.linalg.cholesky(Hp)
        Hp_inv = torch.cholesky_inverse(L)

        Delta_W = None
        if G is not None and beta != 0.0:
            Delta_W = (0.5 * beta * G[:, p]) @ Hp_inv   # [rows, cols]
        
        L_diag = torch.diag(L)
        L = L / L_diag.unsqueeze(0)  # Broadcast division: each column divided by its diagonal
        L = L - torch.eye(L.shape[0], device=L.device)
        del L_diag

        Wr = Wr[:, p] 
        del Hp_inv

        if Delta_W is not None:
            Wr = Wr - Delta_W
            del Delta_W


        Q = torch.zeros_like(Wr)

        g_idx = []
        scale = []
        zero = []
        seen_groups = set()  # Track which groups we've already saved
        
        for i2 in range(self.columns, 0, -blocksize):
            i1 = max(i2 - blocksize, 0)
            count = i2 - i1
            W1 = Wr[:, i1:i2].clone()
            W2diff = Wr[:, i2:] - Q[:, i2:]
            What1 = Q[:, i1:i2].clone()
            L1 = L[:, i1:i2]
            
            for i in reversed(range(count)):
                if groupsize != -1:
                    gstart = (i1 + i) // groupsize * groupsize
                    gend   = min(gstart + groupsize, self.columns)
                    group_id = (i1 + i) // groupsize

                    if group_id not in seen_groups:
                        self.quantizer.find_params(Wr[:, gstart:gend], weight=True)
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        seen_groups.add(group_id)

                What = W1[:,i] + (W1 - What1) @ L1[i1:i2,i] + W2diff @ L1[i2:,i]
                What1[:, i] = self.quantizer.quantize(What.unsqueeze(1)).flatten()
            Q[:, i1:i2] = What1


        Q = Q @ P.t().to(Q.device)
        
        if args.incoh_process:
            Q = incoherence_process(Q, SU, SV, scaleWH, args)
        torch.cuda.synchronize()
        error = torch.trace((W - Q) @ H @ (W - Q).t())

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        # todo
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        self.print_loss(name=name, q_weight=Q, alpha=error.item(), timecost=(time.time() - tick))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        
        # Reverse scale and zero to match original column order since we processed in reverse
        scale = torch.cat(scale[::-1], dim=1)
        zero = torch.cat(zero[::-1], dim=1)
        return scale, zero, g_idx, None


    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
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

