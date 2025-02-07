from abc import ABC, abstractmethod
import argparse
import re
import collections
import os
from glob import glob
import math

# Define an abstract class.
class EnvUtil(ABC):
    scale = 10**6
    EnvStep = 7.5 * scale
    batch_size = 60000
    
    
    def __init__(self, trPath, tePath):
        self.trPath = trPath
        self.tePath = tePath
        
    @abstractmethod
    def parser_init(self):
        pass

    @abstractmethod
    def calc_metric(self):
        pass
    
    @abstractmethod
    def set_batchsize_epoch(self):
        pass
    
    def set_density(self, density):
        self.density = {'map':density[0], 'na':density[1], 'nt':density[2]}
    
    def set_n_agents_by_density(self, args):
        dmap, dna, dnt = self.density['map'], self.density['na'], self.density['nt'],
        r = args.grid_size/dmap
        na = dna*r**2
        nt = dnt*r**2
        
        args.n_agents = int(na)
        args.n_preys = int(nt)
        return args
    
    def set_n_epochs_by_na_EnvSteps(self, args):
        if args.debug != 1:
            args.bs, args.n_epochs = self.set_batchsize_epoch(args)
        return args

    def get_epoch_must_train(self, EnvStep, na, bs):
        return math.ceil((EnvStep * na) / bs)

    def epoch_to_EnvStep(self, epoch, na, bs):
        return (epoch / na * bs)

def add_parser_commons(parser):
    #! common-scenario
    parser.add_argument('--agent_visible', type=int, default=1)
    parser.add_argument('--rendering', type=int, default=0, help='render a scenario or not, 0: No / 1: Yes')

    #! common-loss and range
    # Loss Type
    parser.add_argument('--loss_type', nargs='+', type=int, default=(1, 1, 0), help='(ASSYM, EVERY_GCN, each_transmit)')
    parser.add_argument('--loss_apply', type=int, default=1, help='0: EVERY_ENVSTEP / 1: EVERY_GCN')
    parser.add_argument('--channelType', type=str, default=None,
                        help='FC:fully connection / FL: fully loss / IID: identically independent distrinution loss / GE: gilbert eliot loss')
    parser.add_argument('--InputChannelInfo', type=int, default=0, help='Give channel information to Model or Not')
    
    # Independent and Identically Distributed (IID) loss

    # Gilbert Elliot loss(GE)
    parser.add_argument('--GE_INIT',type=int, default=1, help='1: Good State / 0: Bad State / -1: Proportional init state') 
    parser.add_argument('--Pgb', type=float, default=0.0196, help='')
    parser.add_argument('--Pbg', type=float, default=0.282, help='')
    
    #! common-Architecture
    parser.add_argument('--LSTM', type=int, default=0, help='0: MLP / 1: LSTM')
    parser.add_argument('--lstm_hidden_size', default=64, type=int)
    parser.add_argument('--gcn_layer_update', type=int, default=1, help='GCN layer parameter update or fix')

    # Policy
    # Example: --encoder_hidden_sizes 12 123 1234 
    parser.add_argument('--encoder_hidden_sizes', nargs='+', type=int)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--attention_type', type=str, default='general')
    parser.add_argument('--residual', type=int, default=1)
    parser.add_argument('--categorical_mlp_hidden_sizes', nargs='+', type=int)
    parser.add_argument('--aggregator_type', type=str, default='sum')
    parser.add_argument('--policy_hidden_sizes', nargs='+', type=int)
    parser.add_argument('--hidden_sizes', nargs='+', type=int)

    #! common-Train
    parser.add_argument('--curriculum_learning', type=int, default=0, help='Applying increasing Packet Loss in Training or not')
    parser.add_argument('--n_epochs', type=int, default=1001)
    parser.add_argument('--bs', type=int, default=60000)
    # parser.add_argument('--origin_bs', type=int, default=0)
    parser.add_argument('--n_envs', type=int, default=1)
    # Eval
    parser.add_argument('--run_id', type=int, default=0) # sequential naming
    parser.add_argument('--n_eval_episodes', type=int, default=50)
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--inspect_steps', type=int, default=0)
    parser.add_argument('--eval_during_training', type=int, default=0)
    parser.add_argument('--eval_greedy', type=int, default=1)
    parser.add_argument('--eval_epoch_freq', type=int, default=5)

    #! common-test
    parser.add_argument('--save_test_result', type=int, default=1, help='')

    #! RL Algo
    # parser.add_argument('--max_algo_path_length', type=int, default=n_steps)
    parser.add_argument('--hidden_nonlinearity', type=str, default='tanh')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--center_adv', type=int, default=1)
    parser.add_argument('--positive_adv', type=int, default=0)
    parser.add_argument('--gae_lambda', type=float, default=0.97)
    parser.add_argument('--ent', type=float, default=0.1)
    parser.add_argument('--entropy_method', type=str, default='regularized')
    parser.add_argument('--clip_grad_norm', type=float, default=7)
    parser.add_argument('--opt_n_minibatches', type=int, default=3,
        help='The number of splits of a batch of trajectories for optimization.')
    parser.add_argument('--opt_mini_epochs', type=int, default=10,
        help='The number of epochs the optimizer runs for each batch of trajectories.')

    #! common-file formats
    parser.add_argument('--exe', type=str, default=None, help='model architectures: cent / commDP / obsDP')
    parser.add_argument('--debug', type=int, default=0, help='')
    parser.add_argument('--torch_tensor_type', type=str, default='float', help='default tensor type: float or double')

    parser.add_argument('--device', type=str, default='cpu', help='cpu / cuda:0 / cuda:1 / ...')
    parser.add_argument('--hybrid', type=int, default=0)
    
    parser.add_argument('--cmd', type=str, default='train_test', help='train / test / train_test (sequentially run)')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train / restore / test / eval')
    
    parser.add_argument('--loc', type=str, default='model')
    parser.add_argument('--exp_name', type=str, default=None, help='input pre setted file name OR parameters as argparse form')
    parser.add_argument('--seed', '-s', type=int, default=1)

    parser.add_argument('--n_gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_bias', type=int, default=1)
    
    parser.add_argument('--trRcom', type=int, default=9)
    parser.add_argument('--trpltype', type=str, default='iid')
    parser.add_argument('--trpl', type=float, default=0)
    
    parser.add_argument('--trpf', nargs='+', default=[0, 0, 0, 0], type=float)
    parser.add_argument('--trpftype', type=str, default='iid')
    parser.add_argument('--trpfNmax', type=float, default=0)
    parser.add_argument('--trpfNmin', type=float, default=0)
    parser.add_argument('--trpfPmax', type=float, default=0)
    parser.add_argument('--trpfPmin', type=float, default=0)
    parser.add_argument('--smallMapTrain', type=int, default=0, help='0: Large Map training / 1: default density map training')
    
    parser.add_argument('--teRcom', type=int, default=9)
    parser.add_argument('--tepltype', type=str, default='iid')
    parser.add_argument('--tepl', type=float, default=0)
    
    parser.add_argument('--tepf', nargs='+', default=[0, 0, 0, 0], type=float)
    parser.add_argument('--tepftype', type=str, default='iid')
    parser.add_argument('--tepfNmax', type=float, default=0)
    parser.add_argument('--tepfNmin', type=float, default=0)
    parser.add_argument('--tepfPmax', type=float, default=0)
    parser.add_argument('--tepfPmin', type=float, default=0)
    parser.add_argument('--testScale', type=int, default=0, help='0: TrTeSame / 1: Transfer test')

    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--query', type=str, default=None)

    return parser
    
def get_parser_to_args(parser):
    args = parser.parse_args()
    d = vars(args)
    for k, v in d.items():
        if type(v) == float:
            if float(v) == int(v):
                d[k] = int(v)
    
    if args.categorical_mlp_hidden_sizes is None:
        args.categorical_mlp_hidden_sizes = [128, 64, 32] # Default hidden sizes

    if args.policy_hidden_sizes is None:
        args.policy_hidden_sizes = [128, 64, 32] # Default hidden sizes

    if args.hidden_sizes is None:
        args.hidden_sizes = [128, 64, 32] # Default hidden sizes

    if args.encoder_hidden_sizes is None:
        args.encoder_hidden_sizes = [128, ] # Default hidden sizes

    args.grid_size = args.map
    args.Rsen = args.sen
    
    base_map, base_na, base_nt = 10, int(args.den * 100), int(args.den * 100)
    args.density = [base_map, base_na, base_nt]

    if hasattr(args, 'cap'):
        args.load = args.cap
    
    if args.cmd == 'train':
        args.trpl = args.loss
    elif args.cmd == 'train_test':
        args.trpl = args.tepl = args.loss
    else:
        args.tepl = args.loss
        
    if args.loss == 0:
        args.channelType = 'FC' # fully loss
    elif 0 < args.loss and args.loss < 1.0: 
        args.channelType = 'IID'
    elif args.loss == 1:
        args.channelType = 'FL' # fully connected
    
    return args 

