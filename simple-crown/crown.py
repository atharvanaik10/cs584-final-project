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
        ub = lb = None
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

class SimplexBoundedSequential(nn.Sequential):
    """This class wraps the above BoundedSequential object with simplex bound computation.
    """
    def __init__(self, *args):
        super(SimplexBoundedSequential, self).__init__(*args)

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
        return SimplexBoundedSequential(*layers)


    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        return self.simplex_verify(x_U=x_U, x_L=x_L)

    def simplex_verify(self, x_U=None, x_L=None, num_iters=10):
        modules = list(self._modules.values())
        device  = x_U.device

        # one parameter per neuron
        a, abar = [], []
        for m in modules:
            if isinstance(m, BoundLinear):
                a.append(nn.Parameter(torch.rand(m.out_features, device=device)))
                abar.append(nn.Parameter(torch.rand(m.out_features, device=device)))
            else:
                # placeholder so that a[idx] is defined for every index
                a.append(None)
                abar.append(None)

        opt = torch.optim.Adam([p for p in a + abar if p is not None], lr=1e-2)

        x_U = self.project_simplex(x_U)
        x_L = self.project_simplex(x_L)

        self.attach_interval_bounds(x_L, x_U)

        for _ in range(num_iters):
            opt.zero_grad()
            loss = -self.simplex_backward(modules, a, abar, x_L).mean()
            loss.backward(retain_graph=True)
            opt.step()
            for p in a + abar:
                if p is not None:
                    p.data.clamp_(0.0, 1.0)

        bound = self.simplex_backward(modules, a, abar, x_L)
        return bound, bound
    
    def attach_interval_bounds(self, x_L, x_U):
        """
        Forward interval-bound propagation.
        After the call every BoundLinear / BoundReLU layer owns
        .lower_l  and .upper_u  tensors with shape  (batch, out_features).
        """
        L, U = x_L, x_U
        for layer in self._modules.values():
            if isinstance(layer, BoundLinear):
                W, b = layer.weight, layer.bias
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
                L_next = L @ W_pos.t() + U @ W_neg.t() + b
                U_next = U @ W_pos.t() + L @ W_neg.t() + b
                layer.lower_l, layer.upper_u = L_next, U_next
                L, U = L_next, U_next
            elif isinstance(layer, BoundReLU):
                L, U = L.clamp(min=0), U.clamp(min=0)
                layer.lower_l, layer.upper_u = L, U

    def simplex_backward(self, layers, a, abar, x_L):
        batch = x_L.size(0)
        device = x_L.device

        # Start with identity specification
        A_pos = torch.eye(layers[-1].out_features, device=device).expand(batch, -1, -1)
        A_neg = torch.zeros_like(A_pos)
        bias  = torch.zeros(batch, layers[-1].out_features, device=device)

        for idx in reversed(range(len(layers))):
            layer = layers[idx]

            if isinstance(layer, BoundLinear):
                W, b = layer.weight, layer.bias
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                # interval bounds (already attached by _attach_interval_bounds)
                l_k, u_k = layer.lower_l, layer.upper_u
                theta = (u_k - l_k) / (u_k - l_k + 1e-12)         #   θ_k
                gamma = -l_k                                      #   γ_k

                batch = A_pos.size(0)

                # broadcast a_k  and ā_k
                coef_a    = a[idx].view(1, 1, -1).expand(batch, -1, -1)      # (batch,1,out)
                coef_abar = abar[idx].view(1, 1, -1).expand(batch, -1, -1)

                # broadcast weights
                W_pos_b = W_pos.unsqueeze(0).expand(batch, -1, -1)           # (batch,out,in)
                W_neg_b = W_neg.unsqueeze(0).expand(batch, -1, -1)

                # broadcast θ and γ
                theta_b = theta.unsqueeze(-1)                                # (batch,out,1)
                gamma_b = gamma.unsqueeze(-1)                                # (batch,out,1)

                # affine-map propagation
                A_pos_new = (coef_a    * A_pos) @ W_pos_b + \
                            (coef_abar * A_neg) @ (theta_b * W_pos_b)

                A_neg_new = (coef_a    * A_neg) @ W_neg_b + \
                            (coef_abar * A_pos) @ (theta_b * W_neg_b)

                # broadcast bias vectors per batch
                b_pos_exp = b.clamp(min=0).unsqueeze(0).expand(batch, -1)     # (batch,out)
                b_neg_exp = b.clamp(max=0).unsqueeze(0).expand(batch, -1)     # (batch,out)

                # gamma for Planet upper shift
                gamma_b = gamma.unsqueeze(-1)  # (batch,out,1)

                # bias update – three terms
                bias = bias \
                    + torch.bmm(A_pos, b_pos_exp.unsqueeze(-1)).squeeze(-1) \
                    + torch.bmm(A_neg, b_neg_exp.unsqueeze(-1)).squeeze(-1) \
                    + torch.bmm(coef_abar * A_neg, gamma_b).squeeze(-1)

                A_pos, A_neg = A_pos_new, A_neg_new

            elif isinstance(layer, BoundReLU):
                lb = layer.lower_l.clamp(max=0)
                ub = layer.upper_u.clamp(min=0)
                slope = ub / (ub - lb + 1e-12)
                mask  = (slope > 0.5).float()

                pos = A_pos.clamp(min=0); neg = A_pos.clamp(max=0)
                A_pos = slope * pos + mask * neg

                pos = A_neg.clamp(min=0); neg = A_neg.clamp(max=0)
                A_neg = slope * neg + mask * pos

                inc = torch.bmm(A_pos + A_neg, (-lb * slope).unsqueeze(-1)).squeeze(-1)
                bias = bias + inc

        # concrete lower bound on simplex (worst case = min since A_pos ≥0, A_neg≤0)
        z_min = self.project_simplex(x_L)
        bound = (A_pos + A_neg).bmm(z_min.view(batch, -1, 1)).squeeze(-1) + bias
        return bound

    def project_simplex(self, x):
        v, _ = torch.sort(x, dim=1, descending=True)
        cssv = torch.cumsum(v, dim=1) - 1
        ind  = torch.arange(1, x.size(1) + 1, device=x.device)
        cond = v - cssv / ind > 0
        rho  = cond.sum(dim=1, keepdim=True)
        theta = cssv.gather(1, rho - 1) / rho
        return torch.clamp(x - theta, min=0)
    

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

