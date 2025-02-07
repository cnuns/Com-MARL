import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')
sys.path.append(current_file_path + '/../../')

import re
import numpy as np
from env_uitils import EnvUtil, add_parser_commons, get_parser_to_args
from formats import FileFormat

trPath = f'{current_file_path}/data/model'
teDataRoot = f'{current_file_path}/data/test'
tePath = f'{current_file_path}/data/test/csv'
teDataPath = f'{current_file_path}/data/test/matlab'
teHistoryPath = f'{current_file_path}/data/test/param'
teDataBKPath = f'{current_file_path}/data/test/backup'

if not os.path.isdir(trPath): os.mkdir(trPath)
if not os.path.isdir(teDataRoot): os.mkdir(teDataRoot)
if not os.path.isdir(tePath): os.mkdir(tePath)
if not os.path.isdir(teDataPath): os.mkdir(teDataPath)
if not os.path.isdir(teHistoryPath): os.mkdir(teHistoryPath)
if not os.path.isdir(teDataBKPath): os.mkdir(teDataBKPath)

class PPUtil(EnvUtil, FileFormat):
    def __init__(self, trPath=trPath, tePath=tePath, density=None):
        super().__init__(trPath, tePath)
        if type(density) in [list, tuple]:
            if len(density) == 2:
                self.density = {'map':density[0], 'na':density[1], 'nt':density[2]} 
            else:
                self.density = {'map':density[0], 'na':density[1], 'nt':density[2]} 
        elif type(density) == int:
            raise Exception('Type density as iteratable form')
        
        elif density == None:
            pass
        
        self.num_points = 100
        self.teHistoryPath = teHistoryPath
        self.rScenario = re.compile(f'pp')
        self.scenario_path = os.path.dirname(os.path.abspath(__file__))

    def calc_metric(self, args, trAvgReward, optimal, cntRc, cntRs, cntRm, cntRp, cntRv, cntRv2):
        trainMetric = abs(args.capture_reward) * cntRc -abs(args.step_cost) * cntRs
        optimal = (abs(args.capture_reward)*args.n_preys)
        trainMetric = trainMetric / optimal
        if np.any(trainMetric > 1.0): raise Exception('Metric cannot overcome 1.0, please check \'capture\' and \'step\' count info ')
        return trainMetric
    
    def set_batchsize_epoch(self, args):
        if 30 <= args.grid_size < 40:
            batchs = int(0.5*self.batch_size*args.n_agents/8)
        else:
            batchs = int(self.batch_size*args.n_agents/8)

        if 30 <= args.grid_size:
            train_env_step = 2*self.EnvStep
        else:
            train_env_step = self.EnvStep
        n_epochs = self.get_epoch_must_train(train_env_step, args.n_agents, batchs) + 1

        return batchs, n_epochs

    def parser_init(self, parser):
        parser.add_argument('--map', type=int, default=10)
        parser.add_argument('--sen', type=int, default=1)
        parser.add_argument('--den', type=float, default=0.04)
        parser.add_argument('--cap', type=int, default=2)
        parser.add_argument('--loss', type=float, default=0, help='packet loss prob.')
        
        parser.add_argument('--scenario', type=str, default='pp')
        parser.add_argument('--rewards', nargs='+', default=[10, 0.1, 0, 0], type=float)
        parser.add_argument('--capture_reward', type=float, default=10)
        parser.add_argument('--step_cost', type=float, default=0.1)
        parser.add_argument('--rm', type=float, default=0.0, help='each agent moving reward')
        parser.add_argument('--penalty', type=float, default=0)

        parser.add_argument('--n_agents', '-n', type=int, default=6)
        parser.add_argument('--n_groups', type=int, default=1, help='')
        parser.add_argument('--n_preys', type=int, default=6)
        parser.add_argument('--max_env_steps', type=int, default=200)
        parser = add_parser_commons(parser)
        args = get_parser_to_args(parser)

        return args