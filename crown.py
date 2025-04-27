import torch
import torch.nn as nn
import numpy as np
from model import SimpleNNRelu, SimpleNNHardTanh
from linear import BoundLinear
from relu import BoundReLU
from hardTanh_question import BoundHardTanh
import time
import argparse

import os
import copy
from torch import nn






import sys
from simplex_verify.tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds
from simplex_verify.plnn.simplex_solver.solver import SimplexLP
from simplex_verify.plnn.simplex_solver import utils



def cifar_loaders(batch_size): ### 设置和返回 CIFAR-10 dataset的data loader，该dataset是机器学习中用于图像分类任务的常见基准数据集
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True,   ### 读取model的weight，读取train dataset
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False,      ### 读取测试数据
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=False, pin_memory=True)    ### 为训练和测试这两个dataset创建了两个DataLoader instance
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,  ### 读取
        shuffle=False, pin_memory=True)
    return train_loader, test_loader



def make_elided_models(model, return_error=False):  # 给model换最后一个layer
    elided_models = []
    layers = [lay for lay in model]     # 取出layers
    assert isinstance(layers[-1], nn.Linear)    # 判断是否是linear layer

    net = layers[:-1]   #不要最后一个layer
    last_layer = layers[-1] #存最后一个layer
    nb_classes = last_layer.out_features    # 最后一个layer的features   
    ### 【output的model数量是基于最后一层的out_features数量的，所以就需要使用最后一层的out_feature数量为1的model，也就是simple crown的那个】

    for gt in range(nb_classes):    ### 循环最后一个layer的features;  每次迭代都有不同的last layer，其余相同
        new_layer = nn.Linear(last_layer.in_features,
                              last_layer.out_features-1)    # 对每个feature生成一个last layer

        wrong_weights = last_layer.weight[[f for f in range(last_layer.out_features) if f != gt], :]
        wrong_biases = last_layer.bias[[f for f in range(last_layer.out_features) if f != gt]]

        if return_error:    # 反了个方向
            new_layer.weight.data.copy_(wrong_weights - last_layer.weight[gt])  # 算出来的layer的weight / prediction weight，与ground truth的差值
            new_layer.bias.data.copy_(wrong_biases - last_layer.bias[gt])
        else:
            new_layer.weight.data.copy_(last_layer.weight[gt] - wrong_weights)
            new_layer.bias.data.copy_(last_layer.bias[gt] - wrong_biases)

        layers = copy.deepcopy(net) + [new_layer]   #换上新layer为最后一个layer
        new_elided_model = nn.Sequential(*layers)
        elided_models.append(new_elided_model)
    return elided_models













class BoundedSequential(nn.Sequential):
    r"""This class wraps the PyTorch nn.Sequential object with bound computation."""

    def __init__(self, *args):
        super(BoundedSequential, self).__init__(*args)
        ### 调用超class的构造函数，使 BoundedSequential 能够像普通 nn.Sequential 一样运行

    @staticmethod
    def convert(seq_model):
        ### 将标准PyTorch nn.Sequential模型 转换为 BoundedSequential模型
            # 遍历输入model中的每一层。根据层的类型替换为有bound的版本(BoundLinear、BoundReLU、BoundHardTanh)
        r"""Convert a Pytorch model to a model with bounds.
        Args:
            seq_model: An nn.Sequential module.

        Returns:
            The converted BoundedSequential module.
        """
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
            elif isinstance(l, nn.Hardtanh):
                layers.append(BoundHardTanh.convert(l))
        return BoundedSequential(*layers)





    #def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, X_data=None, y_label=None):
        ### 根据输入bounds 上限x_U 和 下限x_L， 计算网络输出bound的上限和下限 的高级方法
            # 调用full_boundpropogation函数，在有bound层中使用指定的bound计算方法实际计算这些bound
        r"""Main function for computing bounds.
        bound propogation: 用于分析和确保模型抵御对抗攻击的鲁棒性的方法。神经网络的输入被赋予一组边界，表示每个输入特征可以取的最小值和最大值，
                           这模拟了潜在的对抗性攻击。最具挑战性的部分是处理非线性函数，例如激活函数
        Args:
            x_U (tensor): The upper bound of x.
            x_L (tensor): The lower bound of x.
            upper (bool): Whether we want upper bound.
            lower (bool): Whether we want lower bound.
            optimize (bool): Whether we optimize alpha.
        Returns:
            ub (tensor): The upper bound of the final output.
            lb (tensor): The lower bound of the final output.
        """
        ub = lb = None
        ub, lb = self.full_boundpropogation(x_U=x_U, x_L=x_L, upper=upper, lower=lower, X_data=X_data, y_label=y_label)
        return ub, lb

    def full_boundpropogation(self, x_U=None, x_L=None, upper=True, lower=True, X_data=None, y_label=None):
        r"""A full bound propagation. We are going to sequentially compute the
        intermediate bounds for each linear layer followed by a ReLU layer. For each
        intermediate bound, we call self.boundpropogate_from_layer() to do a bound propagation
        starting from that layer.
        Args:
            x_U (tensor): The upper bound of x.
            x_L (tensor): The lower bound of x.
            upper (bool): Whether we want upper bound.
            lower (bool): Whether we want lower bound.
        Returns:
            ub (tensor): The upper bound of the final output.
            lb (tensor): The lower bound of the final output.
        """
        modules = list(self._modules.values())

        ##### 1. Obtain all pre-activation bounds (通过使用CROWN recursively做到)######
        ###################################################################################
        ######### 替换成simplex里的intermediate_net         ？？？？？？？？？？？？？？？
        ######### 不需要用到boundpropogate_from_layer函数了

        """
        
        # CROWN propagation for all layers
        for i in range(len(modules)):   ### 循环所有layers
            # We only need the bounds before a ReLU/HardTanh layer 
            ### 只处理nonlinear的layers
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                ### 在每个nonlinear层之前，都检查前一层是否为BoundLinear层  
                if isinstance(modules[i - 1], BoundLinear): 
                    ### add a batch dimension， 代表从layer输出到bounds的直接mapping；是上一层的output结果
                    newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                    ### 用CROWN计算pre-activation bounds， 从当前non-linear层之前的linear层开始（layer i-1）
                        # 获取这层的输入的上下bounds
                    ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                            start_node=i - 1)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        """

        model = load_network("./models/relu_model.pth")         # load model
        elided_models = make_elided_models(model, True)     # 【output的model数量是基于最后一层的out_features数量的，所以就需要使用最后一层的out_feature数量为1的model，也就是simple crown的那个】
        #_, test_loader = cifar_loaders(1)   ### test_loader是分batch的数据, 每个batch由data和正确label组成
            # 改成: 把data和正确label传过来

        ### simplex代码内把data和label分为好几批, 也就是X和y,  而这里不需要
        cuda_elided_model = copy.deepcopy(elided_models[0]).cuda()  
        intermediate_net = SimplexLP([lay for lay in cuda_elided_model], max_batch=3000) 
        cuda_domain = (X_data.cuda(), eps)      ### X.cuda(), args.eps

        with torch.no_grad():
            intermediate_net.set_solution_optimizer('best_naive_simplex', None)
            ### 存储中间层bounds和weights, 只处理linearlayer
            intermediate_net.define_linear_approximation(cuda_domain, no_conv=False,
                                                         override_numerical_errors=True)
        ### 一系列bounds数据
        intermediate_ubs = intermediate_net.upper_bounds
        intermediate_lbs = intermediate_net.lower_bounds


        lirpa_params = {    
                "nb_outer_iter": 3,
        }
        lirpa_net = SimplexLP(cuda_elided_model, params=lirpa_params,
                             store_bounds_progress=len(intermediate_net.weights), debug=True, dp=True)
        lirpa_net.optimize = lirpa_net.auto_lirpa_optimizer
        lirpa_net.logger = utils.OptimizationTrace()
        lirpa_net.build_model_using_intermediate_net(cuda_domain,
                                                        (intermediate_lbs, intermediate_ubs), 
                                                        intermediate_net)





        ###### 3. Recursively propagate linear inequality y <= aTz + b through each layer ######
        ########################################################################################
        # Get the final layer bound
        """C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U)    """
            # 在α-CROWN中，你用输入的线性函数来表示输出的界限，为此必须通过每一层从输出回溯到输入。
            #  但是你需要一个回溯的起点——那就是：C = torch.eye(output_dim)，它初始化输出约束矩阵
            #  它用 constraint y=y 来初始化 propagation w
        return self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                              C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                                              lower=lower, start_node=i, lirpa_net=lirpa_net)


    def boundpropogate_from_layer(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, lirpa_net=None):
        r"""The bound propagation starting from a given layer. Can be used to compute intermediate bounds or the final bound.

        Args:
            x_U (tensor): The upper bound of x.
            x_L (tensor): The lower bound of x.

            C (tensor): The initial coefficient matrix. Can be used to represent the output constraints.
            But we don't have any constraints here. So it's just an identity matrix.

            upper (bool): Whether we want upper bound.
            lower (bool): Whether we want lower bound.

            start_node (int): The start node of this propagation. It should be a linear layer.
        Returns:
            ub (tensor): The upper bound of the output of start_node.
            lb (tensor): The lower bound of the output of start_node.
        """
        ### modules是一列layers
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node + 1]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])


        ###### 3. Recursively propagate linear inequality y <= aTz + b through each layer ######
        ########################################################################################
        ### 这些方法处理linear inequalities如何通过layer进行转换
            #reversed(modules)意味着反方向遍历layers，从output layer开始
        for i, module in enumerate(reversed(modules)):
            """ ### 替换boundpropogate函数, 变成simplex算法
            bound = self.get_bound_dp_lirpa_backward(weights, additional_coeffs, lower_bounds, upper_bounds)
                # additional_coeffs没用到
                # weights == self.weights 在define_linear_approximation内定下
                # self.lower_bounds和self.upper_bounds   在build_model_using_intermediate_net内,从intermediate_bounds中复制而来
            """
            ### 每个layer，module.boundpropogate(...) 将inequality aT *z + b 转化为新形式
                # 对linear和non-linear layer的操作都在这里, 分别使用relu.py和linear.py内的boundpropogate()函数
                # 带有b的输出都是bias
                # 这个函数是反向地走。所以是 反地在一个反方向的network里走
                # 根据layer类型，前往linear或者relu文件
            upper_A, upper_b, lower_A, lower_b = module.boundpropogate(upper_A, lower_A, start_node, lirpa_net)
            ###############################################
            ######### 也许需要修改relu的boundpropogate函数        ？？？？？？？？？？？？？？？
            # 检查有扰动的时候是否也>0
            
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b


        ###### 4. When the linear inequality propagates to the input layer, we can concretize ######
        ############################################################################################
        # sign = +1: upper bound, sign = -1: lower bound
        def _get_concrete_bound(A, sum_b, sign=-1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            bound = bound.squeeze(-1) + sum_b
            return bound

        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb




eps = 0.1


if __name__ == '__main__':
    # Create the parser
    """parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--activation', default='relu', choices=['relu', 'hardtanh'],
                        type=str, help='Activation Function')
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    # Parse the command line arguments
    args = parser.parse_args()"""

    ### 从 PyTorch.pth文件加载测试数据和labels，文件强制选为“data1.pth”
    x_test, label = torch.load("data1.pth")
    
    #_, test_loader = cifar_loaders(1)
    X_data = x_test.deepcopy()
    y_label = label.deepcopy()



    ### 从SimpleNNRelu中加载model架构，然后从文件中加载其训练参数。
    print('use ReLU model')
    ### 只是定义了 有linear和nonlinear层的 nn.Sequential()
    model = SimpleNNRelu() 
    ### 是PyTorch框架中定义的一种方法，用于将参数和buffer从一个instance加载到  state_dict的一个Module instance及其所有子模块中
    model.load_state_dict(torch.load('models/relu_model.pth'))  
            
    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)

    ### 得到预测结果
    output = model(x_test)
    y_size = output.size(1)
    print("Network prediction: {}".format(output))




    #eps = 0.1           ### 代表l1 Pertubation; global variable      ##############？？？？？？？？？？？？？？？
    x_u = x_test + eps  ### 表示扰动后输入的上下限。
    x_l = x_test - eps

    print(f"Verifiying Pertubation - {eps}")
    start_time = time.time()

    ### 它将标准PyTorch nn.Sequential模型换为 有bound的BoundedSequential模型。   以便之后计算 扰动输入下的输出bound
        # 遍历输入model中的每一层。根据层的类型替换为有bound的版本 (即BoundLinear)  
    boundedmodel = BoundedSequential.convert(model)     




    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, X_data=X_data, y_label=y_label)  ### 计算扰动下模型输出的上下bounds



    ### 遍历每个样本和输出feature，打印每个样本和输出特性所计算的bound，  从而验证对于给定的扰动输入, 模型的输出是否在这些bound之内
    for i in range(batch_size):
        for j in range(y_size):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))






