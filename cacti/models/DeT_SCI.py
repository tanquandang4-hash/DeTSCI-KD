import torch
import torch.nn as nn
from .builder import MODELS
import functools #包含高阶函数的库，可对函数进行处理#
import torch.nn.functional as F
from dcn.modules.deform_conv import * ##
import einops
from einops.layers.torch import Rearrange

class ResBlock_d3d(nn.Module):
    def __init__(self, dim):
        super(ResBlock_d3d, self).__init__()
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.dcn1 = nn.Sequential(
            DeformConvPack_d(dim,dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dimension='THW'),
            #nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
            DeformConvPack_d(dim, dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dimension='THW'),
            #nn.Conv3d(dim, dim, (1, 3, 3), padding=(0, 1, 1)),
        )
    def forward(self, x):
        x = x.contiguous()
        identity = x
        out = self.dcn1(x)
        return out + identity


class TimesAttention3D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., export_attn=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, (dim // 2) * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // 2, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.export_attn = export_attn
        self.last_attn = None

    def forward(self, x):
        B_, N, C = x.shape
        C = C // 2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        if self.export_attn:
            self.last_attn = attn.detach()
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv3d(dim,dim*2,1),
            nn.LeakyReLU(inplace=True)
        )
        self.f1 = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True)
        )
        self.f2 = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True)
        )
        self.mlp = nn.Conv3d(dim*2,dim,1)

    def forward(self,x):
        x = self.p1(x)
        x1,x2 = torch.chunk(x,chunks=2,dim=1)
        f1_out = self.f1(x1)
        f2_out = self.f2(x2*F.gelu(f1_out))
        x = torch.cat([f1_out,f2_out],dim=1)
        out = self.mlp(F.leaky_relu(x))
        return out

class DeTBlock(nn.Module):
    def __init__(self, dim):
        super(DeTBlock, self).__init__()
        self.resblock = ResBlock_d3d(dim)
        self.tsab = TimesAttention3D(dim, num_heads=4)
        self.ffn = FFN(dim)
        '''self.ffn = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, 1)
        )'''

    def forward(self,x):
        _,_,_,h,w = x.shape
        res_out = self.resblock(x)
        tsab_in = einops.rearrange(x,"b c d h w->(b h w) d c")
        tsab_out = self.tsab(tsab_in)
        tsab_out = einops.rearrange(tsab_out,"(b h w) d c->b c d h w",h =h,w=w)
        ffn_in = res_out+tsab_out
        ffn_out = self.ffn(ffn_in)+ffn_in
        return ffn_out


class DenseRNetBlock(nn.Module):
    def __init__(self, dim, group_num):
        super().__init__()
        self.dim = dim
        self.group_num = group_num
        self.split_ratio = 0.5  # 分割比例
        self.part_dim = int(dim * self.split_ratio)  # 处理部分通道数
        self.group_dim = self.part_dim // group_num  # 每组通道数

        # 仅对part1进行分组处理
        self.deblock_list = nn.ModuleList()
        for _ in range(group_num):
            self.deblock_list.append(DeTBlock(self.group_dim))

        # 合并后的通道调整
        self.last_conv = nn.Conv3d(dim, dim, 1)  # 保持输出通道与输入一致

    def forward(self, x):

        part1, part2 = torch.split(x, [self.part_dim, self.dim - self.part_dim], dim=1)

        # 仅对part1进行分组处理
        input_list = torch.chunk(part1, chunks=self.group_num, dim=1)
        out_list = []
        cf_in = input_list[0]
        cf_out = self.deblock_list[0](cf_in)
        out_list.append(cf_out)
        for i in range(1, self.group_num):
            cf_in = input_list[i] + cf_out
            cf_out = self.deblock_list[i](cf_in)
            out_list.append(cf_out)
        processed_part1 = torch.cat(out_list, dim=1)

        # 合并处理后的part1与保留的part2
        out = torch.cat([processed_part1, part2], dim=1)
        out = self.last_conv(out)
        return x + out  # 残差连接
        
class StageBlock(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        group_num=dim//64
        for i in range(depth):
            self.blocks.append(
                DenseRNetBlock(dim,group_num=group_num)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        out = x
        return out

@MODELS.register_module
class DeTSCI(nn.Module):
    def __init__(self, color_channels, dim=128, stage_num=1, depth_num=[2, 1]):
        """
        :param color_channels: 输出图像的通道数
        :param dim: 基础通道数
        """
        super(DeTSCI, self).__init__()
        self.color_channels = color_channels

        # 编码器：通道逐步从1 -> dim -> dim*2
        self.input1 = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
        )
        dim *= 2
        self.encoder = nn.ModuleList()
        for i in range(stage_num):
            self.encoder.append(
                nn.ModuleList([
                    StageBlock(
                        dim=dim, depth=depth_num[i]),
                    nn.Sequential(
                        nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                        nn.LeakyReLU(inplace=True)
                    )

                ])
            )

        self.bottleneck = StageBlock(dim=dim, depth=depth_num[-1])

        self.decoder = nn.ModuleList()
        for i in range(stage_num):
            self.decoder.append(
                nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(dim, dim, 1),
                    ),
                    nn.Conv3d(dim * 2, dim, 1, 1),
                    StageBlock(
                        dim=dim, depth=depth_num[stage_num - 1 - i]
                    ),
                ])
            )

        # 解码器：逐步恢复通道数，最终输出 color_channels
        self.out1 = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, 1),
            Rearrange("b c t h w-> b t c h w"),
            nn.PixelShuffle(2),
            Rearrange("b t c h w-> b c t h w"),
            nn.Conv3d(dim // 2, color_channels, kernel_size=3, stride=1, padding=1),
        )

    def bayer_init(self, y, Phi, Phi_s):
        bayer = [[0, 0], [0, 1], [1, 0], [1, 1]]
        b, f, h, w = Phi.shape
        y_bayer = torch.zeros(b, h // 2, w // 2, 4).to(y.device)
        Phi_bayer = torch.zeros(b, f, h // 2, w // 2, 4).to(y.device)
        Phi_s_bayer = torch.zeros(b, h // 2, w // 2, 4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[..., ib] = y[:, ba[0]::2, ba[1]::2]
            Phi_bayer[..., ib] = Phi[:, :, ba[0]::2, ba[1]::2]
            Phi_s_bayer[..., ib] = Phi_s[:, ba[0]::2, ba[1]::2]
        y_bayer = einops.rearrange(y_bayer, "b h w ba->(b ba) h w")
        Phi_bayer = einops.rearrange(Phi_bayer, "b f h w ba->(b ba) f h w")
        Phi_s_bayer = einops.rearrange(Phi_s_bayer, "b h w ba->(b ba) h w")

        meas_re = torch.div(y_bayer, Phi_s_bayer)
        meas_re = torch.unsqueeze(meas_re, 1)
        maskt = Phi_bayer.mul(meas_re)
        x = meas_re + maskt
        x = einops.rearrange(x, "(b ba) f h w->b f h w ba", b=b)

        x_bayer = torch.zeros(b, f, h, w).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            x_bayer[:, :, ba[0]::2, ba[1]::2] = x[..., ib]
        x = x_bayer.unsqueeze(1)
        return x

    def forward(self, y, Phi, Phi_s):
        out_list = []
        if self.color_channels == 3:
            x = self.bayer_init(y, Phi, Phi_s)
        else:
            meas_re = torch.div(y, Phi_s)
            # meas_re = torch.unsqueeze(meas_re, 1)
            maskt = Phi.mul(meas_re)
            x = meas_re + maskt
            x = x.unsqueeze(1)

        f_x = self.input1(x)
        x = f_x

        fea_list = []
        for stage_block, Downsample in self.encoder:
            x = stage_block(x)
            fea_list.append(x)
            x = Downsample(x)
        x = self.bottleneck(x)

        for i, [Upsample, Fusion, stage_block] in enumerate(self.decoder):
            x = Upsample(x)
            x = Fusion(torch.cat([x, fea_list.pop()], dim=1))
            x = stage_block(x)
        out = self.out1(torch.cat([x, f_x], dim=1))

        if self.color_channels != 3:
            out = out.squeeze(1)
        out_list.append(out)
        return out_list



# Student (lightweight) model
# ------------------------
@MODELS.register_module
class DeTSCIStudent(nn.Module):
    def __init__(self, color_channels, dim=96, stage_num=1, depth_num=[2, 1], num_heads=2, export_attn=True, use_dcn=False):
        super(DeTSCIStudent, self).__init__()
        self.color_channels = color_channels
        self.use_dcn = use_dcn

        # encoder stem: 1 -> dim -> 2*dim with spatial downsampling (stride 2 on H,W)
        self.input1 = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(inplace=True),
        )
        dim *= 2


        class StudentResBlock_d3d(nn.Module):
            def __init__(self, d, use_dcn=False):
                super().__init__()
                self.lrelu = nn.LeakyReLU(inplace=True)
                if use_dcn:
                    # 仅第一层用DCN，第二层用标准卷积
                    self.conv = nn.Sequential(
                        DeformConvPack_d(d, d, kernel_size=(1, 3, 3), stride=1,
                                         padding=(0, 1, 1), dimension='THW'),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv3d(d, d, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                    )
                else:
                    self.conv = nn.Sequential(
                        nn.Conv3d(d, d, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv3d(d, d, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                    )

            def forward(self, x):
                x = x.contiguous()
                identity = x
                out = self.conv(x)
                return out + identity

        class StudentDeTBlock(nn.Module):
            def __init__(self, d, num_heads=2, use_dcn=False):
                super().__init__()
                #self.resblock = StudentResBlock_d3d(d)
                self.resblock = StudentResBlock_d3d(d, use_dcn=use_dcn)
                self.tsab = TimesAttention3D(d, num_heads=num_heads, export_attn=export_attn)
                self.ffn = FFN(d)
            def forward(self, x):
                _, _, _, h, w = x.shape
                res_out = self.resblock(x)
                tsab_in = einops.rearrange(x, "b c d h w->(b h w) d c")
                tsab_out = self.tsab(tsab_in)
                tsab_out = einops.rearrange(tsab_out, "(b h w) d c->b c d h w", h=h, w=w)
                ffn_in = res_out + tsab_out
                ffn_out = self.ffn(ffn_in) + ffn_in
                return ffn_out


        class StudentDenseRNetBlock(nn.Module):
            def __init__(self, d, group_num, use_dcn_first=False):
                super().__init__()
                self.dim = d
                self.group_num = max(1, group_num)
                self.split_ratio = 0.5
                self.part_dim = int(d * self.split_ratio)
                self.group_dim = max(1, self.part_dim // self.group_num)

                self.deblock_list = nn.ModuleList()
                for i in range(self.group_num):
                    # 🔧 默认禁用 DCN 以避免 cuDNN 兼容性问题
                    use_dcn = (i == 0) and use_dcn_first
                    self.deblock_list.append(StudentDeTBlock(self.group_dim, num_heads=2, use_dcn=use_dcn))

                self.last_conv = nn.Conv3d(d, d, 1)

            def forward(self, x):
                part1, part2 = torch.split(x, [self.part_dim, self.dim - self.part_dim], dim=1)
                input_list = torch.chunk(part1, chunks=self.group_num, dim=1)
                out_list = []
                cf_in = input_list[0]
                cf_out = self.deblock_list[0](cf_in)
                out_list.append(cf_out)
                for i in range(1, self.group_num):
                    cf_in = input_list[i] + cf_out
                    cf_out = self.deblock_list[i](cf_in)
                    out_list.append(cf_out)
                processed_part1 = torch.cat(out_list, dim=1)
                out = torch.cat([processed_part1, part2], dim=1)
                out = self.last_conv(out)
                return x + out

        class StudentStageBlock(nn.Module):
            def __init__(self, dim, depth=1, use_dcn_first=False):
                super().__init__()
                self.blocks = nn.ModuleList()
                group_num = max(1, dim // 64)
                for _ in range(depth):
                    self.blocks.append(StudentDenseRNetBlock(dim, group_num=group_num, use_dcn_first=use_dcn_first))
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x

        self.encoder = nn.ModuleList()
        for i in range(stage_num):
            self.encoder.append(
                nn.ModuleList([
                    StudentStageBlock(dim=dim, depth=depth_num[i], use_dcn_first=use_dcn),
                    nn.Sequential(
                        nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                        nn.LeakyReLU(inplace=True)
                    )
                ])
            )

        self.bottleneck = StudentStageBlock(dim=dim, depth=depth_num[-1], use_dcn_first=use_dcn)

        self.decoder = nn.ModuleList()
        for i in range(stage_num):
            self.decoder.append(
                nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(dim, dim, 1),
                    ),
                    nn.Conv3d(dim * 2, dim, 1, 1),
                    StudentStageBlock(dim=dim, depth=depth_num[stage_num - 1 - i], use_dcn_first=use_dcn),
                ])
            )

        self.out1 = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, 1),
            Rearrange("b c t h w-> b t c h w"),
            nn.PixelShuffle(2),
            Rearrange("b t c h w-> b c t h w"),
            nn.Conv3d(dim // 2, color_channels, kernel_size=3, stride=1, padding=1),
        )

    def bayer_init(self, y, Phi, Phi_s):
        bayer = [[0, 0], [0, 1], [1, 0], [1, 1]]
        b, f, h, w = Phi.shape
        y_bayer = torch.zeros(b, h // 2, w // 2, 4).to(y.device)
        Phi_bayer = torch.zeros(b, f, h // 2, w // 2, 4).to(y.device)
        Phi_s_bayer = torch.zeros(b, h // 2, w // 2, 4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[..., ib] = y[:, ba[0]::2, ba[1]::2]
            Phi_bayer[..., ib] = Phi[:, :, ba[0]::2, ba[1]::2]
            Phi_s_bayer[..., ib] = Phi_s[:, ba[0]::2, ba[1]::2]
        y_bayer = einops.rearrange(y_bayer, "b h w ba->(b ba) h w")
        Phi_bayer = einops.rearrange(Phi_bayer, "b f h w ba->(b ba) f h w")
        Phi_s_bayer = einops.rearrange(Phi_s_bayer, "b h w ba->(b ba) h w")

        meas_re = torch.div(y_bayer, Phi_s_bayer)
        meas_re = torch.unsqueeze(meas_re, 1)
        maskt = Phi_bayer.mul(meas_re)
        x = meas_re + maskt
        x = einops.rearrange(x, "(b ba) f h w->b f h w ba", b=b)

        x_bayer = torch.zeros(b, f, h, w).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            x_bayer[:, :, ba[0]::2, ba[1]::2] = x[..., ib]
        x = x_bayer.unsqueeze(1)
        return x

    def forward(self, y, Phi, Phi_s):
        out_list = []
        if self.color_channels == 3:
            x = self.bayer_init(y, Phi, Phi_s)
        else:
            meas_re = torch.div(y, Phi_s)
            maskt = Phi.mul(meas_re)
            x = meas_re + maskt
            x = x.unsqueeze(1)

        f_x = self.input1(x)
        x = f_x

        fea_list = []
        for stage_block, Downsample in self.encoder:
            x = stage_block(x)
            fea_list.append(x)
            x = Downsample(x)
        x = self.bottleneck(x)

        for i, [Upsample, Fusion, stage_block] in enumerate(self.decoder):
            x = Upsample(x)
            x = Fusion(torch.cat([x, fea_list.pop()], dim=1))
            x = stage_block(x)
        out = self.out1(torch.cat([x, f_x], dim=1))

        if self.color_channels != 3:
            out = out.squeeze(1)
        out_list.append(out)
        return out_list

'''if __name__ == "__main__":

    import torch
    from thop import profile

    # 初始化模型
    net = DeTSCIStudent(color_channels=1, dim=96, stage_num=1, depth_num=[2, 1], num_heads=2, export_attn=True).cuda()

    # 示例输入
    batch_size = 1
    frames = 8
    height = 256
    width = 256
    y = torch.randn(batch_size, frames, height, width).cuda()  # 输入 y

    # 示例 Phi 和 Phi_s
    # 假设 Phi 和 Phi_s 的形状为 [batch_size, frames, height, width]
    Phi = torch.randn(batch_size, frames, height, width).cuda()
    Phi_s = torch.randn(batch_size, frames, height, width).cuda()

    # 前向传播
    output = net(y, Phi, Phi_s)  # 传递 Phi 和 Phi_s

    # 计算 FLOPs 和参数量
    flops, params = profile(net, inputs=(y, Phi, Phi_s))
    total = sum([param.nelement() for param in net.parameters()])

    # 打印结果
    print('Number of params: %.2fM' % (total / 1e6))
    print('Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))'''