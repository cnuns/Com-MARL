
import math
import time
import torch
import re
import os
import numpy as np
from garage.experiment.experiment import dump_json
from testing import get_test_range_EnvStep_file
from formats import FileFormat

def ready_to_train(args, scenLib, flag, debug):
    args.mode = 'train'
    
    scenLib.set_density(args.density)
    args = scenLib.set_n_agents_by_density(args)
    args = scenLib.set_n_epochs_by_na_EnvSteps(args)
    
    if args.exp_name is None: # Run by argparse way
        args.exp_name = FileFormat.process_exp_name_default(args)
    else:
        # Run by pre-defined filename
        args = FileFormat.parameter_parsing(args)

    trainForm = FileFormat.getTrainForm(args.exp_name)
    args.original_exp_name = args.exp_name
    args.exp_name = f'{trainForm}'
    
    if debug or args.debug:
        args.n_epochs = 3
    else:
        args.bs, args.n_epochs = scenLib.set_batchsize_epoch(args)

    assert args.max_env_steps*args.n_agents <= args.bs, 'Must be max_env_steps x #of agents <= batch size! set the batch size smaller'

    if args.hybrid:
        args.devices = {'sample':'cpu', 'batch':args.device}
    else:
        args.devices = {'sample':args.device, 'batch':args.device}

    args.flag = flag # training stop flag

    args.torch_tensor_type = 'float'
    if args.torch_tensor_type == 'float':
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.DoubleTensor)

    if 'train' in args.mode or 'restore' in args.mode:
        print('################# TRAINING START with Parameters #################')
        for k, v in vars(args).items():
            print(k, v)
    return args
    
    
def ready_to_test(args, scenLib, debug):
        scenLib.set_density(args.density)
        args = scenLib.set_n_agents_by_density(args)
        args = scenLib.set_n_epochs_by_na_EnvSteps(args)
        
        if 'test' in args.cmd:
            args.mode = 'test'
        elif 'eval' in args.cmd:
            args.mode = 'eval'

        PATH_TEST = './data/test'
        dir_csv = args.dir_csv = f'{PATH_TEST}/csv'
        dir_backup = args.dir_backup = f'{PATH_TEST}/backup'
        dir_matlab = args.dir_matlab =  f'{PATH_TEST}/matlab'
        if not os.path.isdir(dir_backup): os.mkdir(dir_backup)
        if not os.path.isdir(dir_matlab): os.mkdir(dir_matlab)
        if not os.path.exists(dir_csv): os.makedirs(dir_csv)
        args.print_freq = 1
        args.save_freq = 1
        
        if args.exp_name is None:
            args.exp_name = FileFormat.process_exp_name_default(args)
        else:
            if hasattr(args, 'original_exp_name') and args.original_exp_name != args.exp_name:
                args.exp_name = args.original_exp_name

        if debug or args.debug:
            args.n_eval_episodes = 10
            
        args = FileFormat.parameter_parsing(args)
        
        for k, v in vars(args).items():
            print(k, v)
            
        args.model_to_loading = trainForm = FileFormat.getTrainForm(args.exp_name)
        model_path = f'./data/model/{args.model_to_loading}'
        if not os.path.isdir(model_path):
            print('Test setup name: ', args.exp_name)
            print('Train setup name: ', trainForm)
            raise Exception('Attempt to Testing of Not Trained Model')

        if debug or args.debug:
            epoch_list_ = np.array([1,2,3])
            xEnvlist_ = np.array([1,2,3])
        else:
            epoch_list_, xEnvlist_ = get_test_range_EnvStep_file(model_path=model_path,
                                                                 scale=10**6,
                                                                 start=args.start,
                                                                 end=args.end,
                                                                 num_points=args.num_points)
        args.epoch_list = epoch_list_.tolist()
        args.xEnvlist = xEnvlist_.tolist()
        
        for k, v in vars(args).items():
            print(k, v)

        dump_json(f'{scenLib.teHistoryPath}/{args.exp_name}.json', vars(args))
        args.epoch_list, args.xEnvlist = epoch_list_, xEnvlist_
        args.func_parameter_parsing = FileFormat.parameter_parsing
        
        return args
    
