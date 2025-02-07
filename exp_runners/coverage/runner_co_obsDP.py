import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../')
sys.path.append(current_file_path + '/../../')

from custom_implement.utils import set_nn_device_with, check_nn_on_device
parent_folder = os.path.abspath(os.path.join(current_file_path, os.pardir, os.pardir))
sys.path.append(parent_folder)

import argparse
import joblib
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment.deterministic import set_seed
from envs import CoverageWrapper

from com_marl.torch.algos import CentralizedMAPPO
from com_marl.experiment.local_runner_wrapper import LocalRunnerWrapper
from com_marl.sampler import CentralizedMAOnPolicyVectorizedSampler
from com_marl.torch.baselines import CommBaseCritic # baseline
from com_marl.torch.policies import DecCategoricalMLPPolicy # policy


from exp_runners.routine import *
from custom_implement.utils import set_nn_device_with, check_nn_on_device
from custom_implement.randomness import fix_randomness
from exp_runners.testing import *

from eval_co import eval_simple, eval_model
try: from utils_co import *
except: from .utils_co import *

import signal
flag = [0]
def on_exit_signal(signum, frame):
    global flag
    flag[0] = 1
    print("[SubProcess End] Train end by signal")

signal.signal(signal.SIGINT, on_exit_signal)

def obsDP(args, env, ):
    hidden_nonlinearity = F.relu if args.hidden_nonlinearity == 'relu' else torch.tanh

    policy = DecCategoricalMLPPolicy(
        env.spec,
        env.n_agents,
        hidden_nonlinearity=hidden_nonlinearity,
        hidden_sizes=args.policy_hidden_sizes,
        name='dec_categorical_mlp_policy',
        device=args.device,
    )
    
    baseline = CommBaseCritic(
        env.spec,
        n_agents=args.n_agents,
        encoder_hidden_sizes=args.encoder_hidden_sizes,
        embedding_dim=args.embedding_dim,
        attention_type=args.attention_type,
        n_gcn_layers=args.n_gcn_layers,
        residual=bool(args.residual),
        gcn_bias=bool(args.gcn_bias),
        aggregator_type=args.aggregator_type,
        device=args.device,
    )
    
    return policy, baseline


def run(args):
    args = co.preSetting(args, scenario_name='cover')

    if args.mode == 'train':
        # making sequential log dir if name already exists
        @wrap_experiment(name=args.exp_name,
                         prefix=args.prefix,
                         log_dir=args.exp_dir,
                         snapshot_mode='gap_and_last',
                         snapshot_gap=1)
        
        def train_predatorprey(ctxt=None, args_dict=vars(args)):
            args = SimpleNamespace(**args_dict)
            
            set_seed(args.seed)
            fix_randomness(args.seed)

            env = CoverageWrapper(
                centralized=True,
                other_agent_visible=bool(args.agent_visible),
                params = vars(args)
            )
            env = GarageEnv(env)

            runner = LocalRunnerWrapper(
                ctxt,
                eval=args.eval_during_training,
                n_eval_episodes=args.n_eval_episodes,
                eval_greedy=args.eval_greedy,
                eval_epoch_freq=args.eval_epoch_freq,
                save_env=env.pickleable,
            )
            
            policy, baseline = obsDP(args, env)

            # Set max_path_length <= max_steps
            # If max_path_length > max_steps, algo will pad obs
            # obs.shape = torch.Size([n_paths, algo.max_path_length, feat_dim])
            algo = CentralizedMAPPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=args.max_env_steps, # Notice
                discount=args.discount,
                center_adv=bool(args.center_adv),
                positive_adv=bool(args.positive_adv),
                gae_lambda=args.gae_lambda,
                policy_ent_coeff=args.ent,
                entropy_method=args.entropy_method,
                stop_entropy_gradient=True \
                   if args.entropy_method == 'max' else False,
                clip_grad_norm=args.clip_grad_norm,
                optimization_n_minibatches=args.opt_n_minibatches,
                optimization_mini_epochs=args.opt_mini_epochs,
                device=args.device,
            )

            # check neural networks on target device
            check_nn_on_device(algo, args.device)

            runner.setup(algo, env,
                sampler_cls=CentralizedMAOnPolicyVectorizedSampler, 
                sampler_args={'n_envs': args.n_envs},
                hybrid_mode=args.hybrid,
                devices=args.devices,
                flag=args.flag,
                )
            
            runner.train(n_epochs=args.n_epochs, batch_size=args.bs)

        train_predatorprey(args_dict=vars(args))

    elif args.mode in ['restore', 'eval', 'test']:
        exp_dir = model_path = f'{PATH_MODEL}/{args.model_to_loading}'


        if args.mode == 'restore':
            data = joblib.load(exp_dir + '/params.pkl')
            algo = data['algo']
            env = data['env']

            from com_marl.experiment.runner_utils import restore_training
            restore_training(exp_dir, args.exp_name, args, env_saved=env.pickleable, env=env)

        elif args.mode == 'eval':
            data = joblib.load(exp_dir + '/params.pkl')
            algo = data['algo']
            
            set_policy_attributes(algo.policy, args)

            env = CoverageWrapper(
                centralized=True,
                other_agent_visible=bool(args.agent_visible),
                params = vars(args)
            )

            device = args.device
            set_nn_device_with(algo, device, print_option=True)

            eval_simple(args, env, algo)
            
        elif args.mode == 'test':
            env = CoverageWrapper(
                centralized=True,
                other_agent_visible=bool(args.agent_visible),
                params = vars(args)
            )
            
            test_and_output_result_file(args, env, flag, model_path, PATH_TEST, scenLib, eval_model)
            

if __name__ == '__main__':
    co = scenLib = COUtil()
    try: PC_NAME = os.popen('echo %PC_NAME%').read().strip()
    except: PC_NAME = 'temp'
    PATH_MODEL = './data/model'
    PATH_TEST = './data/test'
    
    parser = argparse.ArgumentParser()
    args = co.parser_init(parser)
    args.PC_NAME = PC_NAME
    args.EXEC_Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.exe = 'obs'
    
    # args.debug = 1 # this will start training just 3-epochs and testing for each epochs
    # args.Map = 10
    # args.Sen = 1
    # args.Den = 0.03
    #! Run by
    # python runner_co_obsDP.py --cmd train --map 10 --sen 1 --den 0.03 --loss 0
    if 'train' in args.cmd:
        args = ready_to_train(args, scenLib, flag, debug=0)
        run(args)
    
    flag[0] = 0
    
    if 'test' in args.cmd or 'eval' in args.cmd:
        args = ready_to_test(args, scenLib, debug=0)
        # args.render = 1 # if you want to render the environment while test, add this
        run(args)
    


