import akro
import torch
from torch import nn
import numpy as np

from .mlp_encoder_module import MLPEncoderModule
from .attention_module import AttentionModule
from .graph_conv_module import GraphConvolutionModule
from custom_implement.tensor_slicing import slicing

class CommBaseNet(nn.Module):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,
                 gcn_bias=True,
                 state_include_actions=False,
                 name='comm_base',
                 residual=True,
                 device='cpu',
                ):

        super().__init__()

        self.residual = residual
        self.device = device
        self._n_agents = n_agents
        self.name = name
        self.comm = True
        self.centralized = True
        self.step = 0
        self.eps = 1e-12
        self._cent_obs_dim = env_spec.observation_space.flat_dim
        self._dec_obs_dim = int(self._cent_obs_dim / n_agents)
        if isinstance(env_spec.action_space, akro.Discrete): 
            self._action_dim = env_spec.action_space.n
        else:
            self._action_dim = env_spec.action_space.shape[0]
        self._embedding_dim = embedding_dim

        self.n_gcn_layers = n_gcn_layers

        self.layers = []

        if state_include_actions:
            self._dec_obs_dim += self._action_dim
        
        self.encoder = MLPEncoderModule(input_dim=self._dec_obs_dim,
                                        output_dim=self._embedding_dim,
                                        hidden_sizes=encoder_hidden_sizes,
                                        output_nonlinearity=torch.tanh)
        self.layers.append(self.encoder)


        self.attention_layer = AttentionModule(dimensions=self._embedding_dim, 
                                                attention_type=attention_type)
        self.layers.append(self.attention_layer)

        self.gcn_layers = nn.ModuleList([GraphConvolutionModule(in_features=self._embedding_dim,
                                                                out_features=self._embedding_dim,
                                                                bias=gcn_bias,
                                                                id=i) for i in range(self.n_gcn_layers)])
        self.layers.append(self.gcn_layers)

        for layer in self.layers:
            layer.to(device)

        
    def grad_norm(self):
        # temp = []
        # for i, p in enumerate(self.parameters()):
        #     temp.append(p.grad.norm(2).item() ** 2)
        # temp = [p.grad.norm(2).item() ** 2 for p in self.parameters()]

        return np.sqrt(np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, obs_n, Range, channels, get_actions):
        # Partially decentralize, treating agents as being independent
        # (n_paths, max_path_length, n_agents, emb_feat_dim)
        # or (n_agents, emb_feat_dim)

        """
            obs_n: (1, n, dim_obs)
            embeddings_0: (1,n,d) /
            Range: (1,n,n) / 
            channels: (1,n_gcn,n,n) / 
        """
    
        embeddings_collection = []
        E = self.encoder.forward(obs_n)

        embeddings_collection.append(E)
        attention_weights = M = self.attention_layer.forward(E)
        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            #* trick: for advantage of computation power,
            #* first, we get attention weights assuming fully connected and no loss
            #* second, manipulate attention weights doing element-wise dot-product with communication range, loss
            L_i = slicing(channels, -3, i_layer)
            A_i = M * Range * L_i
            A_i = A_i / (A_i.sum(dim=-1, keepdim=True) + self.eps) # renormalization

            embeddings_gcn = gcn_layer.forward(inputs=embeddings_collection[i_layer], A=A_i , n_gcn_layer=len(self.gcn_layers))
            embeddings_collection.append(embeddings_gcn)
            
        return embeddings_collection, attention_weights

    def reset(self, dones):
        return