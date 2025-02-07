import akro
import torch
from torch import nn
import numpy as np
import copy

from torch.distributions import Categorical
from com_marl.torch.modules import CategoricalLSTMModule, CommBaseNet


class CommCategoricalLSTMPolicy(CommBaseNet):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 gcn_bias=True,
                 lstm_hidden_size=64,
                 state_include_actions=False,
                 name='comm_categorical_lstm_policy',
                 device='cpu',
                 ):
        
        print('##### Actor Net: LSTM')
        
        assert isinstance(env_spec.action_space, akro.Discrete), (
            'Categorical policy only works with akro.Discrete action space.')

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
            state_include_actions=state_include_actions,
            name=name,
            device=device,
        )
        
        self.device = device
        self.residual = residual
        self.state_include_actions = state_include_actions

        self._prev_actions = None
        self._prev_hiddens = None
        self._prev_cells = None
        
        # Policy layer
        self.categorical_lstm_output_layer = \
            CategoricalLSTMModule(input_size=self._embedding_dim,
                                  output_size=self._action_dim,
                                  hidden_size=lstm_hidden_size).to(device)
        self.layers.append(self.categorical_lstm_output_layer)

    # Batch forward
    def forward(self, obs_n, avail_actions_n, dist_adj, channels, actions_n=None):
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        avail_actions_n = avail_actions_n.reshape(avail_actions_n.shape[:-1] + (self._n_agents, -1))
        size = channels.shape[:-2] + (len(self.gcn_layers), self._n_agents, self._n_agents)

        channels = channels.reshape(size)
        if dist_adj.shape[-2:] != torch.Size((self._n_agents, self._n_agents)):
            size = dist_adj.shape[:-1] + (self._n_agents, self._n_agents)
            dist_adj =  dist_adj.reshape(size)


        n_paths = obs_n.shape[0]
        max_path_len = obs_n.shape[1]
        if self.state_include_actions:
            assert actions_n is not None
            actions_n = torch.Tensor(actions_n).unsqueeze(-1).type(torch.LongTensor)
            # actions_n.shape = (n_paths, max_path_len, n_agents, 1)
            # Convert actions_n to one hot encoding
            actions_onehot = torch.zeros(actions_n.shape[:-1] + (self._action_dim,))
            # actions_onehot.shape = (n_paths, max_path_len, n_agents, action_dim)
            actions_onehot.scatter_(-1, actions_n, 1)
            # Shift and pad actions_onehot by one time step
            actions_onehot_shifted = actions_onehot[:, :-1, :, :]
            # Use zeros as _prev_actions in the first time step
            zero_pad = torch.zeros(n_paths, 1, self._n_agents, self._action_dim)
            # Concatenate zeros to the beginning of actions
            actions_onehot_shifted = torch.cat((zero_pad, actions_onehot_shifted), dim=1)
            # Combine actions into obs
            obs_n = torch.cat((obs_n, actions_onehot_shifted), dim=-1)

        embeddings_collection, attention_weights = super().forward(obs_n, dist_adj, channels, get_actions=False)

        if self.residual:
            inputs = embeddings_collection[0] + embeddings_collection[-1] # applying_skip_connection
        else:
            inputs = embeddings_collection[-1]
            
        # inputs.shape = (n_paths, max_path_len, n_agents, emb_dim) 
        inputs = inputs.transpose(0, 1)
        # inputs.shape = (max_path_len, n_paths, n_agents, emb_dim)
        inputs = inputs.reshape(max_path_len, n_paths * self._n_agents, self._embedding_dim)

        dists_n = self.categorical_lstm_output_layer.forward(inputs)[0]

        # Apply available actions mask
        masked_probs = dists_n.probs.reshape(max_path_len, n_paths, self._n_agents, self._action_dim)
        masked_probs = masked_probs.transpose(0, 1)

        masked_probs = masked_probs * avail_actions_n # mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dists_n = Categorical(probs=masked_probs) # redefine distribution

        return masked_dists_n, attention_weights

    def step_forward(self, obs_n, avail_actions_n, dist_adj, channels, get_actions=True):
        """
            Single step forward for stepping in envs
        """
        if get_actions:
            obs_n = torch.Tensor(obs_n).to(self.device)
            obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
            avail_actions_n = avail_actions_n.reshape(avail_actions_n.shape[:-1] + (self._n_agents, -1))

            dist_adj = torch.Tensor(dist_adj).to(self.device)
            channels = torch.Tensor(channels).to(self.device)

        n_envs = obs_n.shape[0] if len(obs_n.shape) > 2 else 1
        if self.state_include_actions:
            if self._prev_actions is None:
                self._prev_actions = torch.zeros(n_envs, self._n_agents, self._action_dim)
            obs_n = torch.cat((obs_n, self._prev_actions), dim=-1)

        embeddings_collection, attention_weights = super().forward(obs_n, dist_adj, channels, get_actions)

        if self.residual:
            inputs = embeddings_collection[0] + embeddings_collection[-1] # applying_skip_connection
        else:
            inputs = embeddings_collection[-1]
            

        inputs = inputs.reshape(1, n_envs * self._n_agents, self._embedding_dim)

        dists_n, next_h, next_c = self.categorical_lstm_output_layer.forward(inputs, self._prev_hiddens, self._prev_cells)

        self._prev_hiddens = next_h
        self._prev_cells = next_c

        masked_probs = dists_n.probs.reshape(n_envs, self._n_agents, self._action_dim)
        masked_probs = dists_n.probs.cpu() * torch.Tensor(avail_actions_n) # mask 
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dists_n = Categorical(probs=masked_probs) # redefine distribution

        return masked_dists_n, attention_weights.cpu()
    
    def get_actions(self, obs_n, avail_actions_n, dist_adj, channels, greedy=False):
        """Independent agent actions (not using an exponential joint action space)
            
        Args:
            obs_n: list of obs of all agents in ONE time step [o1, o2, ..., on]
            E.g. 3 agents: [o1, o2, o3]

        """
        with torch.no_grad():
            dists_n, attention_weights = self.step_forward(obs_n, avail_actions_n, dist_adj, channels, get_actions=True)
            if not greedy:
                actions_n = dists_n.sample().numpy()
            else:
                actions_n = np.argmax(dists_n.probs.numpy(), axis=-1)
            agent_infos_n = {}
            agent_infos_n['action_probs'] = [dists_n.probs[i].numpy() 
                for i in range(len(actions_n))]
            agent_infos_n['attention_weights'] = [attention_weights.numpy()[i, :]
                for i in range(len(actions_n))]

            if self.state_include_actions:
                # actions_onehot.shape = (n_envs, self._n_agents, self._action_dim)
                actions_onehot = torch.zeros(len(obs_n), self._n_agents, self._action_dim)
                actions_onehot.scatter_(
                    -1, torch.Tensor(actions_n).unsqueeze(-1).type(torch.LongTensor), 1)     
                self._prev_actions = actions_onehot

            return actions_n, agent_infos_n

    def reset(self, dones):
        if all(dones): # dones is synched
            self._prev_actions = None
            self._prev_hiddens = None
            self._prev_cells = None

    def entropy(self, observations, avail_actions, dist_adj, channels, actions):
        # print('obs.shape =', observations.shape)
        dists_n, _ = self.forward(obs_n=observations,
                                avail_actions_n=avail_actions,
                                dist_adj=dist_adj,
                                channels=channels,
                                actions_n=actions,
        )
        
        # print('dist =', dists_n)
        # print('dist.probs =', dists_n.probs)
        entropy = dists_n.entropy()
        # print('entropy.shapeBefore =', entropy.shape)
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        # print('entropy.shapeAfter =', entropy.shape)
        return entropy

    def log_likelihood(self, observations, avail_actions, dist_adj, channels, actions):
        if self.state_include_actions:
            dists_n, _ = self.forward(observations, avail_actions, actions)
        else:
            dists_n, _ = self.forward(observations, avail_actions, dist_adj, channels)
        llhs = dists_n.log_prob(actions)
        # llhs.shape = (n_paths, max_path_length, n_agents)
        # For n agents action probability can be treated as independent
        # Pa = prob_i^n Pa_i
        # log(Pa) = sum_i^n log(Pa_i)
        llhs = llhs.sum(axis=-1) # Asuming independent actions
        # llhs.shape = (n_paths, max_path_length)
        return llhs

    @property
    def recurrent(self):
        return True
    

