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

    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
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
        return self.full_boundpropogation(x_U=x_U, x_L=x_L, upper=upper, lower=lower, optimize=optimize)

    def full_boundpropogation(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
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
        if optimize:
            ans = self.simplex_verify(x_U=x_U, x_L=x_L)
            return ans, ans
        else:
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

    def simplex_verify(self, x_U, x_L, num_iters=3, lr=0.1):
        """
        Runs projected gradient ascent on the dual weights (a_list, abar_list),
        calling simplex_backward each iteration, and returns the final lower bound.
        """
        # 1) Collect BoundLinear layers
        modules = list(self._modules.values())
        # skip last linear layer since it has no activation
        linear_layers = [m for m in modules if isinstance(m, BoundLinear)][:-1]

        # 2) Initialize dual vectors in [0,1]
        a_list    = [torch.empty(layer.out_features, device=x_U.device).uniform_(0,1).requires_grad_(True) 
                     for layer in linear_layers]
        abar_list = [torch.empty(layer.out_features, device=x_U.device).uniform_(0,1).requires_grad_(True)
                     for layer in linear_layers]

        optimizer = torch.optim.Adam(a_list + abar_list, lr=lr)

        # 3) Gradient ascent loop
        for _ in range(num_iters):
            lb = self.simplex_backward(x_U, x_L, a_list, abar_list)
            loss = -lb.mean()        # maximize mean lower bound
            optimizer.zero_grad()
            loss.backward()
            # project back into [0,1]
            with torch.no_grad():
                for v in a_list + abar_list:
                    v.clamp_(0, 1)

        # 5) Return tightened lower bound
        return self.simplex_backward(x_U, x_L, a_list, abar_list)
    
    def simplex_backward(self, x_U, x_L, a_list, abar_list):
        modules = list(self._modules.values())
        batch_size = x_U.size(0)
        out_dim = modules[-1].out_features

        # Initialize the final affine spec f(x) = A x + b
        A = torch.eye(out_dim, device=x_U.device).unsqueeze(0).repeat(batch_size,1,1)
        b = torch.zeros(batch_size, out_dim, device=x_U.device)

        for m in reversed(modules):
            if isinstance(m, BoundLinear):
                # boundpropogate returns (uA, ubias, lA, lbias)
                # we only care about uA and ubias here
                A, ubias, _, _ = m.boundpropogate(A, None)
                # now shift b by that bias
                b = b + ubias
                break

        # Find all the linear‐layer indices in order, skip last one
        lin_indices = [i for i,m in enumerate(modules) if isinstance(m, BoundLinear)][:-1]
        # We'll consume a_list/abar_list in reverse:
        li = len(lin_indices) - 1

        # Walk backwards over the linear layers but drop last one
        for i in reversed(lin_indices):
            linear = modules[i]
            relu = modules[i+1]  # the very next module is that layer's ReLU

            a = a_list[li]
            abar = abar_list[li]
            li -= 1

            # Zero‐input for bias computation
            x0 = torch.zeros(batch_size, linear.in_features, device=x_U.device)

            # Jacobians at x0:
            def upper_fn(x): 
                # planet bound from ReLU + convex‐hull bound
                return ( abar.view(1,-1) * relu.upper_u + 
                        (1-abar).view(1,-1) * linear.simplex_hull(x) )
            def lower_fn(x):
                return a.view(1,-1) * linear(x)
            
            # Bias terms:
            ubias = upper_fn(x0)
            lbias = lower_fn(x0)

            if ubias.shape[1] == out_dim:  
                # now ubias, lbias, b all are (batch, out_dim)==(batch,10)
                b = ubias + lbias + b

            full_Ju = torch.autograd.functional.jacobian(upper_fn, x0, vectorize=True)
            full_Jl = torch.autograd.functional.jacobian(lower_fn, x0, vectorize=True)
            full_Ju = full_Ju.permute(0,2,1,3)
            full_Jl = full_Jl.permute(0,2,1,3)
            idx = torch.arange(batch_size, device=x0.device)
            Ju = full_Ju[idx, idx]   # picks Ju[k] = full_Ju[k,k,:,:]
            Jl = full_Jl[idx, idx]

            # Decompose A into pos/neg and apply:
            A_pos = A.clamp(min=0)
            A_neg = A.clamp(max=0)
            A = torch.bmm(A_neg, Ju) + torch.bmm(A_pos, Jl)

        # Finally, minimize over x∈Δ by taking the minimum over the vertices:
        min_vert = A.min(dim=2).values
        return b + min_vert


class SimplexBoundedSequential(BoundedSequential):
    @staticmethod
    def convert(seq_model):
        # convert all layers as before
        bs = BoundedSequential.convert(seq_model)

        # simplex-propagation: for each BoundLinear except the last,
        # divide its weights/bias by alpha and multiply the next 
        # linear layer by alpha
        modules = list(bs._modules.values())
        for i, m in enumerate(modules[:-1]):
            if isinstance(m, BoundLinear):
                a = m.alpha
                # down-scale this layer
                m.weight.data.div_(a)
                m.bias.data.div_(a)
                # up-scale the next linear layer
                nxt = modules[i+1]
                if isinstance(nxt, BoundLinear):
                    nxt.weight.data.mul_(a)
        return bs
    

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
        ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=False)
    else:
        print('use simplex-verify algorithm')
        boundedmodel = SimplexBoundedSequential.convert(model)
        ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=True)

    for i in range(batch_size):
        for j in range(y_size):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

