import torch
import torch.nn as nn
from DRN_LwU.model import common


def make_model(opt):
    return DRN(opt)


class DRN(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(DRN, self).__init__()
        self.opt = opt
        self.reg_lambda = opt.reg_lambda
        self.reg_params = None
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3

        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), opt.n_colors, kernel_size)
            )
        self.tail = nn.ModuleList(tail)

        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        # upsample x to target sr size
        x = self.upsample(x)

        # preprocess
        x = self.sub_mean(x)
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)

            results.append(sr)

        return results

    def init_reg_params(self):
        reg_params = {}

        for name, param in self.named_parameters():

            print("Initializing omega values for layer", name)
            omega = torch.zeros(param.size())

            init_val = param.data.clone()
            param_dict = {}

            # for first task, omega is initialized to zero
            param_dict['omega'] = omega
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

        self.reg_params = reg_params

    def init_reg_params_across_tasks(self):
        reg_params = self.reg_params

        for name, param in self.named_parameters():
            print("Initializing omega values for layer for the new task", name)
            # 获取该参数的字典
            param_dict = reg_params[param]

            # Store the previous values of omega
            prev_omega = param_dict['omega']

            # Initialize a new omega
            new_omega = torch.zeros(param.size())

            init_val = param.data.clone()

            # 多加了一个前Ω的属性
            param_dict['prev_omega'] = prev_omega
            # 再把当前任务的Ω初始化为0
            param_dict['omega'] = new_omega

            # 存储参数在前一个任务训练后保留下来的值，作为该任务的初始值
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

        self.reg_params = reg_params

    @property
    def name(self):
        return (
            'DRN'
            '-reg_lambda{reg_lambda}'
        ).format(
            reg_lambda=self.reg_lambda,
        )

    def init_params(self):
        params = []
        for param in self.parameters():
            params.append(param.detach())
        return params

    def get_params(self):
        params = []
        for param in self.parameters():
            params.append(param)
        return params