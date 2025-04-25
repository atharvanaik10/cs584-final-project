import torch
import torch.nn as nn
import numpy as np
from model import SimpleNNRelu
from linear import BoundLinear
from relu import BoundReLU
import time
import argparse


class BoundedSequential(nn.Sequential):
    r"""This class wraps the PyTorch nn.Sequential object with bound computation."""

    def __init__(self, *args):
        super(BoundedSequential, self).__init__(*args)

    @staticmethod
    def convert(seq_model):
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
        return BoundedSequential(*layers)

    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False, simplex_verify=False):
        r"""Main function for computing bounds.

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
        if (simplex_verify):
            ub, lb = self.full_boundpropogation_sv(x_U=x_U, x_L=x_L, upper=upper, lower=lower)
        else:
            ub, lb = self.full_boundpropogation(x_U=x_U, x_L=x_L, upper=upper, lower=lower)
        return ub, lb

    def full_boundpropogation(self, x_U=None, x_L=None, upper=True, lower=True):
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
        # CROWN propagation for all layers
        for i in range(len(modules)):
            # We only need the bounds before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                if isinstance(modules[i - 1], BoundLinear):
                    # add a batch dimension
                    newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                            start_node=i - 1)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # Get the final layer bound
        return self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                              C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                                              lower=lower, start_node=i)

    def boundpropogate_from_layer(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None):
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
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node + 1]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.boundpropogate(upper_A, lower_A, start_node)
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b

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

class SimplexBoundedSequential(BoundedSequential):
    """This class wraps the above BoundedSequential object with simplex bound computation.
    """
    def __init__(self, *args):
        super(SimplexBoundedSequential, self).__init__(*args)

    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        return self.compute_bounds_simplex_verify(x_U=x_U, x_L=x_L)

    def compute_bounds_simplex_verify(self, x_U=None, x_L=None, num_iters=10):
        modules = list(self._modules.values())
        n_layers = len(modules)
        a = [torch.nn.Parameter(torch.rand(1), requires_grad=True) for _ in range(n_layers)]
        abar = [torch.nn.Parameter(torch.rand(1), requires_grad=True) for _ in range(n_layers)]
        opt = torch.optim.Adam(a + abar, lr=0.01)

        x_U = self.simplex_projection(x_U)
        x_L = self.simplex_projection(x_L)

        for _ in range(num_iters):
            opt.zero_grad()
            loss = self.simplex_backward(modules, a, abar, x_U, x_L)
            (-loss).backward()
            opt.step()
            for param in a + abar:
                param.data.clamp_(0, 1)

        final_loss = self.simplex_backward(modules, a, abar, x_U, x_L)
        return final_loss, final_loss

    def simplex_backward(self, modules, a, abar, x_U, x_L):
        batch_size = x_U.shape[0]
        n_layers = len(modules)
        device = x_U.device

        f_pos = torch.eye(modules[-1].out_features, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        f_neg = torch.zeros_like(f_pos)
        f_cst = torch.zeros((batch_size, modules[-1].out_features), device=device)

        for idx in reversed(range(n_layers)):
            layer = modules[idx]
            if isinstance(layer, BoundLinear):
                w = layer.weight
                b = layer.bias

                alpha = layer.simplex_alpha()

                def u_k(x):
                    return torch.relu(nn.functional.linear(x, w, b)) / alpha

                def u_k_prime(x):
                    terms = []
                    for i in range(w.shape[1]):
                        ei = torch.zeros_like(x)
                        ei[:, i] = 1.0
                        y_i = torch.relu(nn.functional.linear(ei, w, b)) - torch.relu(nn.functional.linear(torch.zeros_like(x), w, b))
                        terms.append(x[:, i:i+1] * y_i)
                    return sum(terms) + torch.relu(nn.functional.linear(torch.zeros_like(x), w, b)) / alpha

                upper_term = abar[idx] * u_k(x_L) + (1 - abar[idx]) * u_k_prime(x_L)
                lower_term = a[idx] * (nn.functional.linear(x_L, w, b) / alpha)

                f_new_neg = f_neg.bmm(upper_term.unsqueeze(-1)).squeeze(-1)
                f_new_pos = f_pos.bmm(lower_term.unsqueeze(-1)).squeeze(-1)
                f_new_cst = f_cst

                f = f_new_neg + f_new_pos + f_new_cst
                f_pos = f.unsqueeze(-1)
                f_neg = torch.zeros_like(f_pos)
                f_cst = torch.zeros_like(f)

            elif isinstance(layer, BoundReLU):
                continue

        return f.sum(dim=1).mean()

    def simplex_projection(self, x):
        x = torch.clamp(x, min=0)
        s = torch.sum(x, dim=1, keepdim=True)
        return x / torch.maximum(s, torch.ones_like(s))
    

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='crown', choices=['crown', 'simplex-verify'],
                        type=str, help='Algorithm')
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    # Parse the command line arguments
    args = parser.parse_args()

    x_test, label = torch.load(args.data_file)

    model = SimpleNNRelu()
    model.load_state_dict(torch.load('models/relu_model.pth'))

    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)
    output = model(x_test)
    y_size = output.size(1)
    print("Network prediction: {}".format(output))

    eps = 0.01
    x_u = x_test + eps
    x_l = x_test - eps

    print(f"Verifiying Pertubation - {eps}")
    start_time = time.time()

    if args.algorithm == 'crown':
        print('use default CROWN')
        boundedmodel = BoundedSequential.convert(model)
    else:
        print('use simplex-verify algorithm')
        boundedmodel = SimplexBoundedSequential.convert(model)
    
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)

    for i in range(batch_size):
        for j in range(y_size):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

