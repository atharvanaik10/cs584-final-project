import torch
import torch.nn as nn
import numpy as np
from model import SimpleNNRelu, SimpleNNHardTanh
from linear import BoundLinear
from relu import BoundReLU
from hardTanh_question import BoundHardTanh
import time
import argparse





def boundpropogate_simplex(self, last_uA, last_lA):
    lb_r = self.lower_l.clamp(max=0)
    ub_r = self.upper_u.clamp(min=0)
    ub_r = torch.max(ub_r, lb_r + 1e-8)

    upper_d = ub_r / (ub_r - lb_r)  # slope
    upper_b = - lb_r * upper_d      # intercept
    upper_d = upper_d.unsqueeze(1)

    lower_d = (upper_d > 0.5).float()

    uA = lA = None
    ubias = lbias = 0

    if last_uA is not None:
        pos_uA = last_uA.clamp(min=0)
        neg_uA = last_uA.clamp(max=0)

        uA = upper_d * pos_uA + lower_d * neg_uA    ###
        ubias = pos_uA.matmul(upper_b.unsqueeze(-1)).squeeze(-1)    ###

    if last_lA is not None:
        pos_lA = last_lA.clamp(min=0)
        neg_lA = last_lA.clamp(max=0)

        lA = upper_d * neg_lA + lower_d * pos_lA
        lbias = neg_lA.matmul(upper_b.unsqueeze(-1)).squeeze(-1)

    return uA, ubias, lA, lbias



"""
alpha-CROWN算法用于计算神经网络输出的边界。它通过在网络各层传播约束（线性不等式）来运行，包括 ReLU 等非线性激活函数。
输入：神经网络输入的下限和上限(输入变量可以取的min和max)，网络layers(包括linear和ReLU)
用途：处理处理non-linear layers，保这些layers的输出保持在先前linear layer设置的bound内。通过根据激活函数是“激活”还是“非激活”状态来来调整bound，来做的
输出：在给定输入bound的情况下网络可以产生的max和min的蔬菜

Simplex Verify算法：验证神经网络鲁棒性的方法



l1 norm & l1 ball:
https://www.youtube.com/watch?v=FiSy6zWDfiA
https://www.youtube.com/watch?v=It2g7sDxdqI

norm: 描述vector的size
vector: 可以用一组数字表示的东西， 在ML所有东西都是用vector表示的
     - 例如: vector[3,4] 可以在二位坐标上表示, x=3 & y=4
            l1 norm = vector坐标与原点的 曼哈顿距离
            l2 norm = vector坐标与原点的 直线距离
            ln norm = vector坐标与原点的 直线距离
"""


class SimplexVerifyOptimizer:
    def __init__(self, model, t_max, eps):
        self.model = model
        self.t_max = t_max
        self.eps = eps

    def optimize(self, input, label):
        a = torch.rand(..., requires_grad=True)  # shape depends on # layers
        abar = torch.rand(..., requires_grad=True)

        optimizer = torch.optim.Adam([a, abar], lr=0.01)

        for t in range(self.t_max):
            loss = self.simplex_backward(a, abar, input, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Project a, abar to [0, 1]
            with torch.no_grad():
                a.clamp_(0, 1)
                abar.clamp_(0, 1)

        return loss


