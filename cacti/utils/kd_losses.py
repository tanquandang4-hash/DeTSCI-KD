import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_builder import LOSSES


class FeatureAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = None

    def forward(self, x, out_channels):
        in_channels = x.shape[1]
        if in_channels == out_channels:
            return x
        if self.adapter is None:
            self.adapter = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.adapter.to(x.device)
        elif self.adapter.in_channels != in_channels or self.adapter.out_channels != out_channels:
            # re-init if shape changes across calls
            self.adapter = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.adapter.to(x.device)
        return self.adapter(x)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x, y):
        diff2 = (x - y) * (x - y)
        return torch.mean(torch.sqrt(diff2 + self.eps2))


@LOSSES.register_module
class KDLoss(nn.Module):
    def __init__(self, lambda_out=1.0, lambda_feat=0.5, lambda_attn=0.1, loss_type: str = 'mse'):
        super().__init__()
        self.lambda_out = float(lambda_out)
        self.lambda_feat = float(lambda_feat)
        self.lambda_attn = float(lambda_attn)
        # selectable base loss
        loss_type = (loss_type or 'mse').lower()
        if loss_type == 'smoothl1':
            self.loss_fn = nn.SmoothL1Loss()
        elif loss_type == 'charbonnier':
            self.loss_fn = CharbonnierLoss()
        else:
            self.loss_fn = nn.MSELoss()
        self.adapters = nn.ModuleDict()  # per feature key

    def adapt(self, key, x, target):
        if key not in self.adapters:
            self.adapters[key] = FeatureAdapter()
        adapter = self.adapters[key]
        return adapter(x, target.shape[1])

    def attn_align_and_loss(self, s_list, t_list):
        if len(s_list) == 0 or len(t_list) == 0:
            return torch.tensor(0.0, device=s_list[0].device if len(s_list) > 0 else t_list[0].device)
        n = min(len(s_list), len(t_list))
        loss = 0.0
        for i in range(n):
            s = s_list[i]
            t = t_list[i]
            # both [B, heads, N, N] -> reduce head dim by mean for alignment across different heads
            if s.dim() == 4:
                s = s.mean(dim=1, keepdim=True)
            if t.dim() == 4:
                t = t.mean(dim=1, keepdim=True)
            loss = loss + self.loss_fn(s, t)
        return loss / n

    def forward(self, student_out, teacher_out, feat_s: dict, feat_t: dict, attn_s: list, attn_t: list, scales: dict = None):
        # output distillation
        l_out = self.loss_fn(student_out, teacher_out)

        # feature distillation on intersected keys
        keys = [k for k in feat_s.keys() if k in feat_t]
        l_feat = 0.0
        if len(keys) > 0:
            for k in keys:
                s = feat_s[k]
                t = feat_t[k]
                if s.shape[1] != t.shape[1]:
                    s = self.adapt(k, s, t)
                l_feat = l_feat + self.loss_fn(s, t)
            l_feat = l_feat / len(keys)
        else:
            # no features provided
            l_feat = torch.tensor(0.0, device=student_out.device)

        # attention distillation
        l_attn = self.attn_align_and_loss(attn_s, attn_t)

        # optional per-term scaling for scheduling
        if scales is None:
            s_out = 1.0
            s_feat = 1.0
            s_attn = 1.0
        else:
            s_out = float(scales.get('out', 1.0))
            s_feat = float(scales.get('feat', 1.0))
            s_attn = float(scales.get('attn', 1.0))

        kd = (self.lambda_out * s_out) * l_out + (self.lambda_feat * s_feat) * l_feat + (self.lambda_attn * s_attn) * l_attn
        return {
            'kd': kd,
            'l_out': l_out.detach(),
            'l_feat': l_feat.detach(),
            'l_attn': l_attn.detach(),
        }

