import akro
import torch
from torch import nn
import numpy as np
import copy

from torch.distributions import Normal
from com_marl.torch.modules.gaussian_mlp_module import GaussianMLPModule
from com_marl.torch.modules.comm_base_net import CommBaseNet

class CommBaseCritic(CommBaseNet):

    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 decoder_hidden_sizes=(64, ),
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 gcn_bias=True,
                 share_std=False,
                 state_include_actions=False,
                 aggregator_type='sum',
                 name='base_critic',
                 device='cpu',
                ):

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            residual=residual,
            gcn_bias=gcn_bias,
            state_include_actions=state_include_actions,
            name=name,
            device=device,
        )

        self.device = device
        self.aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            aggregator_input_dim = embedding_dim
        elif aggregator_type == 'direct':
            aggregator_input_dim = embedding_dim * self._n_agents

        self.baseline_aggregator = GaussianMLPModule(
            input_dim=aggregator_input_dim,
            output_dim=1,
            hidden_sizes=decoder_hidden_sizes,
            hidden_nonlinearity=torch.tanh,
            share_std=True,
            ).to(device)

    def compute_loss(self, obs_n, returns, dist_adj, channels, get_actions=False):
        if get_actions:
            obs_n = torch.Tensor(obs_n).to(self.device)
            obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
            dist_adj = torch.Tensor(dist_adj).to(self.device)
            channels = torch.Tensor(channels).to(self.device)
        else:
            obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
            size = channels.shape[:-2] + (len(self.gcn_layers), self._n_agents, self._n_agents)
            channels = channels.reshape(size)
            
            if dist_adj.shape[-2:] != torch.Size((self._n_agents, self._n_agents)):
                size = dist_adj.shape[:-1] + (self._n_agents, self._n_agents)
                dist_adj =  dist_adj.reshape(size)
                
        embeddings_collection, attention_weights = super().forward(obs_n, dist_adj, channels, get_actions)
        if self.residual:
            embeddings_add = embeddings_collection[0] + embeddings_collection[-1]
        else:
            embeddings_add = embeddings_collection[-1]
            
        # shared std, any std = std.mean()
        if self.aggregator_type == 'sum':
            mean, std = self.baseline_aggregator(embeddings_add) # n_epi, 200, 6, 1 / 200, 6, 1
            baseline_dist = Normal(mean.squeeze(-1).sum(-1), std.mean()) # n_epi,200, 
        elif self.aggregator_type == 'direct':
            embeddings_add = embeddings_add.reshape(embeddings_add.shape[:-2] + (-1, )) # concatenate embeddings
            mean, std = self.baseline_aggregator(embeddings_add)
            baseline_dist = Normal(mean.squeeze(-1), std.mean())
        ll = baseline_dist.log_prob(returns)
        return -ll.mean() # baseline loss

    def forward(self, obs_n, avail_actions_n, dist_adj, channels, get_actions=False):
        if get_actions:
            obs_n = torch.Tensor(obs_n).to(self.device)
            obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
            dist_adj = torch.Tensor(dist_adj).to(self.device)
            channels = torch.Tensor(channels).to(self.device)
        else:
            obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
            size = channels.shape[:-2] + (len(self.gcn_layers), self._n_agents, self._n_agents)
            channels = channels.reshape(size)
            
            if dist_adj.shape[-2:] != torch.Size((self._n_agents, self._n_agents)):
                size = dist_adj.shape[:-1] + (self._n_agents, self._n_agents)
                dist_adj =  dist_adj.reshape(size)

        embeddings_collection, attention_weights = super().forward(obs_n, dist_adj, channels, get_actions)
        if self.residual:
            embeddings_add = embeddings_collection[0] + embeddings_collection[-1] # applying_skip_connection
        else:
            embeddings_add = embeddings_collection[-1]

        if self.aggregator_type == 'sum':
            mean, _ = self.baseline_aggregator(embeddings_add)
            return mean.squeeze(-1).sum(-1)
        elif self.aggregator_type == 'direct':
            embeddings_add = embeddings_add.reshape(embeddings_add.shape[:-2] + (-1, ))
            mean, _ = self.baseline_aggregator(embeddings_add)
            return mean.squeeze(-1)

    def get_attention_weights(self, obs_n):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        _, attention_weights = super().forward(obs_n)
        return attention_weights