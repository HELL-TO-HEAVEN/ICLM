import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda')
class ICLMModel(nn.Module):
    def __init__(self, n, T, L, N, tau_1=10, tau_2=0.2, use_gpu=False):
        super(ICLMModel, self).__init__()
        self.T = T
        self.L = L
        self.N = N
        self.n = n
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.w = nn.Parameter(torch.Tensor(self.T - 1, self.L, self.n + 1))
        nn.init.kaiming_uniform_(self.w.view(self.n + 1, -1), a=np.sqrt(5))
        # nn.init.orthogonal_(self.w.view(self.n + 1, -1), gain=1)
        self.weight = nn.Parameter(torch.Tensor(self.L, 1))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        self.bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.bias)

        self.h = torch.ones(self.T - 1, self.L, self.n + 2)
        self.h_y = torch.ones(self.L, self.n + 2)
        self.h = nn.Parameter(self.h)
        self.h_y = nn.Parameter(self.h_y)

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(0.3)

    def forward(self, input, input_all, all_states, t, relation_tail, flag, alpha=0, is_training=False):
        s = 0
        if t != self.T - 1:
            w_probs = self.w[t]
            h_probs = self.h[t]
            h_y_probs = self.h_y
            if flag:
                w_probs = self.w[self.T - 2 - t]
                h_y_probs = self.h[-1]
                if t == self.T - 2:
                    h_probs = self.h_y
                else:
                    h_probs = self.h[self.T - 3 - t]
            w_probs = torch.softmax(w_probs, dim=-1)
            hidden = torch.sparse.sum(input_all, dim=-1)
            hidden = self.activation(hidden.to_dense().view(self.N, self.n + 1)).to_sparse()
            h_probs = self.activation(h_probs / self.tau_1)
            h_probs[:, -2] = 0
            # if is_training and not flag and t > 0: h_probs = torch.cat([h_probs[:, :-1], w_probs[:, -1:]], dim=-1)
            hidden = self.activation(torch.sparse.mm(hidden, torch.permute(h_probs[:, :-1], (1, 0))))
            hidden = hidden * (1 - h_probs[:, -1].unsqueeze(dim=0)) + torch.ones_like(hidden) * h_probs[:, -1].unsqueeze(dim=0)
            hidden_y = input.to_dense().view(self.N, -1, self.n + 1)
            hidden_y = self.activation(torch.sum(hidden_y, dim=0))
            hidden_y = torch.cat([hidden_y[:, int(self.n / 2): -1], hidden_y[:, :int(self.n / 2)], hidden_y[:, -1:]], dim=-1)
            h_y_probs = self.activation(h_y_probs / self.tau_1)
            h_y_probs[:, -2] = 0
            hidden_y = self.activation(torch.mm(hidden_y, torch.permute(h_y_probs[:, :-1], (1, 0))))
            hidden_y = hidden_y * (1 - h_y_probs[:, -1].unsqueeze(dim=0)) + \
                       torch.ones_like(hidden_y) * h_y_probs[:, -1].unsqueeze(dim=0)

            if flag: w_probs = torch.cat([w_probs[:, int(self.n / 2): -1], w_probs[:, :int(self.n / 2)], w_probs[:, -1:]],
                                dim=-1)
            if t == 0:
                w = w_probs
                s_tmp = torch.sparse.mm(input, torch.permute(w, (1, 0))).view(self.N, -1, self.L)
                s = s_tmp * hidden.unsqueeze(dim=1) * hidden_y.unsqueeze(dim=0)
                if is_training: s = self.dropout(s)
            if t >= 1:
                w = w_probs
                s_tmp = torch.sparse.mm(input_all, all_states[t - 1].reshape(self.N, -1))
                s_tmp = s_tmp.view(self.N, self.n + 1, -1, self.L)
                if is_training: s_tmp = self.dropout(s_tmp)
                s_tmp = torch.einsum('mrnl,lr->mnl', s_tmp, w)
                s = s_tmp * hidden.unsqueeze(dim=1)
        else:
            s = all_states[t - 1]
            if is_training: s = self.dropout(s)
            s = torch.squeeze(torch.einsum('nml,lk->nmk', s, torch.tanh(self.weight)), dim=-1)
        return s

    def negative_loss(self, p_score, n_score):
        y = torch.autograd.Variable(torch.Tensor([1]))
        if self.use_gpu: y = y.to(device)
        # return self.criterion(p_score, n_score, y)
        loss = torch.square(1 - torch.minimum(p_score, torch.ones_like(p_score))) + torch.square(torch.maximum(n_score, torch.zeros_like(n_score)))
        return loss


    def log_loss(self, p_score, label, logit_mask, thr=1e-20):
        one_hot = F.one_hot(torch.LongTensor([label]), p_score.shape[-1])
        if self.use_gpu:
            one_hot = one_hot.to(device)
            logit_mask = logit_mask.to(device)
        p_score = p_score - 1e30 * logit_mask.unsqueeze(dim=0)
        loss = -torch.sum(
            one_hot * torch.log(torch.maximum(F.softmax(p_score / self.tau_2, dim=-1), torch.ones_like(p_score) * thr)),
            dim=-1)
        return loss

    def constrain_loss(self, weight, lammda=0.1):
        con = 0
        for t in range(self.T - 1):
            w_ = torch.softmax(self.w[t], dim=-1)
            con += torch.sum((1 - w_) * w_, dim=-1)
        loss = 0
        loss += lammda * con
        return loss

    def gt_constrain(self, x):
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu: zero = zero.to(device)
        return torch.square(torch.minimum(x, zero))

    def lt_constrain(self, x):
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu: zero = zero.to(device)
        return torch.square(torch.maximum(x, zero))

    def eq_constrain(self, x):
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu: zero = zero.to(device)
        return torch.square(torch.maximum(x, zero) + torch.minimum(x, zero))

    def activation(self, x):
        one = torch.autograd.Variable(torch.Tensor([1]))
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu:
            one = one.to(device)
            zero = zero.to(device)
        return torch.minimum(torch.maximum(x, zero), one)
