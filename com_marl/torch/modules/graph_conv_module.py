import sys
import os
# Compute the upper-level folder path.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# Add the upper-level folder path to the import path.
sys.path.append(parent_dir)
# Enable importing modules or packages from the upper-level folder.

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter

def save_data(name, data):
    import pickle
    with open(f'./{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
        
class GraphConvolutionModule(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, id=None):
        super().__init__()
        """
            drop_option: first / except_last / all
                means:
                    first: dropout apply on first layer only
                    except_last: dropout apply on all layer, except_last layer
                    first: dropout apply on all layer, including last layer
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.id = id
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.x_weight = self.weight.clone().detach()
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, A, n_gcn_layer):
        """
            inputs: H^(l): feature matrix
            A: GCN attention weight-parameters
        """
        if len(inputs.shape) == 2:  # when eval mode
            H_l = torch.mm(inputs, self.weight)
            H_l_next = outputs = torch.mm(A, H_l)

        elif len(inputs.shape) > 2:
            # n_paths, max_path_length, n_agents, emb_feat_dim = inputs.size()
            # M_a dim = (n_paths, max_path_length, n_agents, n_agents)
            H_l = torch.matmul(inputs, self.weight) # [1,24,64] x [64x64]
            # support dim = (n_paths, max_path_length, n_agents, emb_feat_dim)
            H_l_next = outputs = torch.matmul(A, H_l) # [1,24,24] x [1,24,64]

        if self.bias is not None:
            outputs = torch.tanh(outputs + self.bias)
        else:
            outputs = torch.tanh(outputs)
            
        return outputs
