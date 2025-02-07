import torch
import numpy as np
import random
import re
import networkx as nx

try: from .gilbert_elliot_loss_model import *
except: from gilbert_elliot_loss_model import *

def init_communication(slf, params):
    slf.eval_n_epi = None
    slf.GCNHops = params.get("n_gcn_layers")
    slf.curriculum_learning = params.get("curriculum_learning")
    slf.success = 0
    slf.ave_trput = 0
    slf.dist_adj = None
    slf.channels = None
    slf.ave_deg = 0
    slf.diameter = 0
    slf.calc_diameter = params.get("calc_diameter", False)
    slf.loss_apply = params.get("loss_apply") # each EnvStep or each GCN layer
    
    slf.mode = params.get("mode")
    
    # set loss probability
    if slf.mode in ['train', 'restore']:
        pref = 'tr'
    else:
        pref = 'te'
    slf.pl = params.get(f"{pref}pl")
    if slf.pl == None:
        raise Exception('Loss probability is not applied. Check exp_name or verify if the regex is working correctly.')
    slf.pconn = 1 - slf.pl

    slf.channelType = params.get("channelType") # FC or FL or IID or GE
    if slf.pl == 0:
        slf.channelType = 'FC' # fully loss
    elif 0 < slf.pl and slf.pl < 1.0: 
        slf.channelType = 'IID'
    elif slf.pl == 1:
        slf.channelType = 'FL' # fully connected
    else:
        raise Exception(f'invalid Ploss value: pl={slf.pl} type={type(slf.pl)}')


    #* GE loss
    if 'GE' in slf.channelType:
        # slf.GE_INIT = 1 # 1: Good / 0: proportional
        slf.Pgb, slf.Pbg = params.get("Pgb"), params.get("Pbg")
        slf.bad_proportion = slf.Pgb / (slf.Pgb+slf.Pbg)
        if slf.Pbg != 0:
            slf.expected_burst_length = 1/slf.Pbg

        slf.GE_INIT = params.get("GE_INIT")
        if slf.GE_INIT == 1:
            init_state = STATE['Good']
        elif slf.GE_INIT == 0:
            init_state = STATE['Bad']
        else:
            init_state = -1

        slf.expected_switch_count = get_GE_num_mean_state_changes(slf.Pgb, slf.Pbg, n=1000, init_state=init_state, Tmax=slf.Tmax)

    # set communication range
    if slf.mode in ['train', 'restore']:
        slf.Rcom = params.get("trRcom")
    else:
        slf.Rcom = params.get("teRcom")
        slf.n_eval_episodes = params.get("n_eval_episodes")
        
    if slf.Rcom+1 >= slf.maps:
        slf.Rcom = 0 # Fully connected
    x = torch.FloatTensor([[0,0], [slf.Rcom,slf.Rcom]])
    distances = torch.cdist(x, x)
    slf.Rcom_th = distances[0][-1]
    
    slf.agent_pos = {_: None for _ in range(slf.n_agents)}

import numpy as np
from collections import Counter
def split_fair_loss_ratio(n_eval_episodes:int, loss_types:list, ):
    fair_loss = loss_types*(n_eval_episodes // len(loss_types))
    leftovers = (n_eval_episodes % len(loss_types))
    for le in range(leftovers):
        idx = le % len(loss_types)
        fair_loss.append(loss_types[idx])

    print('Loss type counts: ', Counter(fair_loss))
    return fair_loss

def update_communication_state(slf):
    slf.dist_adj, slf.ave_deg, slf.diameter = get_graph(slf.Rcom, slf.Rcom_th, slf.n_agents, slf.agent_pos, slf.calc_diameter)
    if slf.channelType == 'FC':
        slf.channels = torch.ones((slf.GCNHops, slf.n_agents, slf.n_agents)).float().numpy()
        return
    
    elif slf.channelType == 'FL':
        ch = torch.eye(slf.n_agents).unsqueeze(0).expand(slf.GCNHops, slf.n_agents, slf.n_agents)
        slf.channels = ch.float().numpy()
        return
    
    if 'IID' in slf.channelType:
        slf.channels = get_iid_channel(depth=slf.GCNHops, n_agents=slf.n_agents, Ploss=slf.pl)
        return

    elif slf.channelType == 'GE':
        if slf._step_count == 0:
            if slf.loss_apply == 0: # loss apply: 0=EnvStep / 1=GCN Layer
                if slf.GE_INIT == STATE['Good']:
                    channels = torch.ones(size=(slf.n_agents, slf.n_agents))
                    slf.state = channels.expand(slf.GCNHops, *channels.shape)
                    slf.channels = slf.state.float().numpy()

                elif slf.GE_INIT == STATE['Bad']:
                    channels = torch.zeros(size=(slf.n_agents, slf.n_agents))
                    slf.state = channels.expand(slf.GCNHops, *channels.shape)
                    slf.channels = slf.state.float().numpy()

                else:
                    channels = get_init_state(n=slf.n_agents, Pgb=slf.Pgb, Pbg=slf.Pbg)
                    slf.state = channels.expand(slf.GCNHops, *channels.shape).unsqueeze(0)
                    slf.channels = slf.state.float().numpy()
            else:
                if slf.GE_INIT == STATE['Good']:
                    channels = torch.ones(size=(slf.n_agents, slf.n_agents))
                    next_state = get_next_state_matrix(n_sequence=slf.GCNHops-1, state=channels.bool(), Pgb=slf.Pgb, Pbg=slf.Pbg, include_prev=True)
                    slf.state = next_state
                    slf.channels = slf.state.float().numpy()

                elif slf.GE_INIT == STATE['Bad']:
                    channels = torch.zeros(size=(slf.n_agents, slf.n_agents))
                    next_state = get_next_state_matrix(n_sequence=slf.GCNHops-1, state=channels.bool(), Pgb=slf.Pgb, Pbg=slf.Pbg, include_prev=True)
                    slf.state = next_state
                    slf.channels = slf.state.float().numpy()

                else:
                    channels = get_init_state(n=slf.n_agents, Pgb=slf.Pgb, Pbg=slf.Pbg)
                    next_state = get_next_state_matrix(n_sequence=slf.GCNHops-1, state=channels.bool(), Pgb=slf.Pgb, Pbg=slf.Pbg, include_prev=True)
                    slf.state = next_state
                    slf.channels = slf.state.float().numpy()
                    
            slf.state_history = [channels]
        else:
            if slf.loss_apply == 0: # loss apply: 0=EnvStep / 1=GCN Layer
                next_state = get_next_state_matrix(n_sequence=1, state=slf.state[-1].bool(), Pgb=slf.Pgb, Pbg=slf.Pbg)

                slf.state = next_state.expand(slf.GCNHops, *next_state.shape[1:])
                slf.channels = slf.state.float().numpy()
                slf.state_history.append(next_state[-1])

            else: # loss different at each GCN layer
                # raise NotImplementedError
                next_state = get_next_state_matrix(n_sequence=slf.GCNHops, state=slf.state[-1].bool(), Pgb=slf.Pgb, Pbg=slf.Pbg)

                slf.state = next_state
                slf.channels = slf.state.float().numpy()
                slf.state_history.append(next_state)

def print_env_params(slf, option):
    if option == True:
        class_variables = slf.__dict__
        print('\n\n############### Env Params Applyed ###############')
        for k, v in class_variables.items():
            if type(v) == list:
                if len(v) > 100:
                    continue
            print(f'{k}: {v}')
        print('\n\n#########################################################\n\n')


def get_euclidean_distance(x):
    if type(x) != torch.Tensor:
        x = torch.FloatTensor(x)

    distances = torch.cdist(x, x)
    return distances

def update_grid_adjacency(pos, rng, dist_adj, i_agent, _grid_shape, _full_obs):
    if rng % 2 == 1:
        dy = dx = int(rng/2)
        for row in range(max(0, pos[0] - dy), min(pos[0] + dy + 1,_grid_shape[0])):
            for col in range(max(0, pos[1] - dx), min(pos[1] + dx + 1, _grid_shape[1])):
                if ('A' in _full_obs[row][col]):
                    # print(re.findall(r'\d+',  _full_obs[row][col]))
                    agent_idx = int(re.findall(r'\d+', _full_obs[row][col])[0])
                    dist_adj[i_agent][agent_idx-1] = 1

    else:
        dy = dx = int(rng/2) - 1
        for row in range(max(0, pos[0] - (dy+1)), min(pos[0] + dy + 1, _grid_shape[0])):
            for col in range(max(0, pos[1] - (dx+1)), min(pos[1] + dx + 1, _grid_shape[1])):
                if ('A' in _full_obs[row][col]):
                    # print(re.findall(r'\d+',  _full_obs[row][col]))
                    agent_idx = int(re.findall(r'\d+', _full_obs[row][col])[0])
                    dist_adj[i_agent][agent_idx-1] = 1

    return dist_adj


def get_iid_channel(depth, n_agents, Ploss):
    """
    identically and independently distribution
    """
    
    """
    this "greater than or equal"( >= ) condition is very important 
    torch.rand in [0, 1) --> rarely sample exactly zero value ( 0. )
    if Ploss = 0, then channel always should be 1
    but if we compate with ">", this can be "Ploss > 0." --> false -> channel value is 0
    """
    I = torch.eye(n_agents).unsqueeze(0).expand((depth, n_agents, n_agents)) # (l, n, n)
    channel_p = torch.rand(size=(depth, n_agents, n_agents)) + I # (l, n, n)
    new_channel = (channel_p >= Ploss).float().numpy()
    return new_channel
        


def get_graph(Rcom, Rcom_th:float, n_agents:int, agent_pos:dict, calc_diameter=False):
    if Rcom == 0: # case0: Fully Conn.
        dist_adj = np.ones((n_agents, n_agents))
        ave_deg = n_agents
        diameter = n_agents
        return dist_adj, ave_deg, diameter

    if type(agent_pos[0]) != torch.Tensor:
        pos = torch.Tensor(list(agent_pos.values()))
    else:
        pos = torch.stack(list(agent_pos.values()))
    distances = get_euclidean_distance(pos)
    dist_adj = (distances <= Rcom_th).float().numpy()
    
    ave_deg = dist_adj.sum(axis=1).mean(axis=0)
        
    diameter = 0
    if calc_diameter:
        G = nx.from_numpy_matrix(dist_adj)
        # Needs to be modified to count subgraphs later.
        if(nx.is_connected(G)):
            diameter = nx.diameter(G)
        else:
            diameter = 0

    return dist_adj, ave_deg, diameter

def graph_Euclidean(pos, threshold):
    """
        Description: Uses Euclidean distance to measure inter-agent distance in a matrix format, improving computational speed.
        
        distance-threshold: "Rcom/2 * sqrt(2)"
            Rcom: full-map covering squared box length
            
            e.g. maps=10 --> Rcom=20 --> threshold=10√2
    """
    distances = get_euclidean_distance(pos)
    A = (distances <= threshold).float()
    return A

def graph_grid(n_agents, dist_adj, agent_pos, agent_rcom, _grid_shape, _full_obs):
    for i_agent in range(n_agents):
        rcom = agent_rcom[i_agent]
        if rcom != 0:
            pos = agent_pos[i_agent]
            dist_adj = update_grid_adjacency(pos=pos, rng=rcom, dist_adj=dist_adj, i_agent=i_agent, _grid_shape=_grid_shape, _full_obs=_full_obs)

        else: #* If agent_i can see entire field
            dist_adj[i_agent] = np.ones((1, n_agents))

    return dist_adj

#* ##################### delays #####################
def delays_init(adjacency, link_loss, delay_th):
    # after spawn, if never seen some nodes because they're far from here
    # -> set delay = threshold + 1  -> I don't care that nodes 
    delays = [np.where(adjacency==0, delay_th, 1)]
    for i, link in enumerate(link_loss[1:]):
        delays.append(np.where(link==0, delays[i]+1, 1))
        
    #! new_delays = self.calc_delays(adjacency=adjacency, link_loss=link_loss, old_delays=delays)
    return np.array(delays)

def calc_delays(adjacency, link_loss, old_delays):
    loss = adjacency * link_loss
    delays = [np.where(loss[0]==0, old_delays+1, 1)]
    for i, l in enumerate(loss[1:]):
        delays.append(np.where(l==0, delays[i]+1, 1))
    return np.array(delays)
    

#* ##################### fault #####################
def iid_fault(n_agents, p_fault):
    agent_condition = np.random.choice([0,1], size=n_agents, p=[p_fault, 1-p_fault])
    return agent_condition

def GE_fault(agent_condition, p, r):
    G_condition = np.where(agent_condition==1)
    B_condition = np.where(agent_condition==0)

    new_condition = np.zeros_like(agent_condition)
    new_condition[G_condition] = np.random.choice([1,0],size=(len(G_condition)) ,p=[1-p, p])
    new_condition[B_condition] = np.random.choice([1,0],size=(len(B_condition)) ,p=[r, 1-r])
    return new_condition

def get_one_hot_vectors(values):
    onehots = []
    for i, elem in enumerate(values):
        temp = [0] * len(values)
        temp[i] = 1
        onehots.append(temp)

    return onehots

def get_broadcast_reliability(D, P):
    """
        Args:
            D: Adjacency matrix of within comm range, size:(n, n)
            P: Link loss probability matrix of all nodes, size:(n, n)
        Return:
            B: broad link reliability matrix about neighboring node set within communication range
    """
    N_i = D * P
    #N_i = N_i - torch.eye(N_i.size(-1))
    cardinalityOf_N_i = torch.sum(N_i > 0, dim=-1, keepdim=True)
    sumOf_r_ij = torch.sum(N_i, dim=-1, keepdim=True)
    B = sumOf_r_ij / (cardinalityOf_N_i + 1e-7)
    
    return B

def estimate_link_prob(Ploss, na, isSameLinkProb=True, estm_algo=False):
    if estm_algo:
        raise NotImplementedError
        # NOTE: here goes channel estimation algorithms .. etc
            # return: matrix of probability of each link

    else:
        if isSameLinkProb:
            assert 0<= Ploss <= 1
            P = torch.full(fill_value=Ploss, size=(na,na))

        else:
            assert type(Ploss) == list
            assert len(Ploss) == na and len(Ploss[0]) == na
            P = torch.Tensor(Ploss)
    
    P = P - torch.eye(na)
    P = torch.clip(P, min=0., max=1.)
    return P