import torch
import torch.nn as nn


class BoundLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BoundLinear, self).__init__(in_features, out_features, bias)

    @staticmethod
    def convert(linear_layer):
        r"""Convert a nn.Linear object into a BoundLinear object

        Args: 
            linear_layer (nn.Linear): The linear layer to be converted.

        Returns:
            l (BoundLinear): The converted layer
        """
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        l.weight.data.copy_(linear_layer.weight.data)
        l.weight.data = l.weight.data.to(linear_layer.weight.device)
        l.bias.data.copy_(linear_layer.bias.data)
        l.bias.data = l.bias.to(linear_layer.bias.device)
        l.alpha = l.simplex_alpha()
        return l
    
    def simplex_alpha(self):
        """Get rescaling coefficient alpha to ensure outputs of linear layer remain simplex

        From Section 4.1 in the simplex-verify paper
        """
        device = self.weight.device
        in_dim = self.weight.shape[1]
        max_sum = 0
        for i in range(in_dim):
            e_i = torch.zeros((1, in_dim), device=device)
            e_i[0, i] = 1
            z = torch.relu(nn.functional.linear(e_i, self.weight, self.bias))
            max_sum = max(max_sum, z.sum().item())
        
        zero_vertex_sum = torch.relu(self.bias).sum().item()
        return max(max_sum, zero_vertex_sum)
    
    def simplex_hull(self, x):
        """
        Compute y_CH(x) = sum_j x_j [ReLU(w^T e_j + b) - ReLU(b)] + ReLU(b)
        where e_j is the j-th standard basis vector.
        x: (batch, in_features)
        returns: (batch, out_features)
        """
        # h0_k = ReLU(b_k)
        h0 = torch.relu(self.bias)                           # (out_features,)

        # preactivation on each vertex e_j: w_kj + b_k
        #   self.weight: (out_features, in_features)
        pre = self.weight + self.bias.unsqueeze(1)           # (out, in)
        h_e = torch.relu(pre)                                # (out, in)

        # difference term ∆_{kj} = h_e[k,j] - h0[k]
        delta = h_e - h0.unsqueeze(1)                        # (out, in)

        # now x @ delta^T + h0
        #   x: (batch, in), delta^T: (in, out)
        y = x.matmul(delta.t()) + h0.unsqueeze(0)            # (batch, out)
        return y

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        r"""Bound propagate through the linear layer

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation. (It's not used in linear layer)


        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.
        """

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            # propagate A to the nest layer
            next_A = last_A.matmul(self.weight)
            # compute the bias of this layer
            sum_bias = last_A.matmul(self.bias)
            return next_A, sum_bias

        uA, ubias = _bound_oneside(last_uA)
        lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias
