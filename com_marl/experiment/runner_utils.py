import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

from custom_implement.utils import set_nn_device_with

from garage.experiment.experiment import ExperimentContext
from garage.experiment.deterministic import set_seed
from com_marl.experiment.local_runner_wrapper import LocalRunnerWrapper
# from tensorboardX import SummaryWriter
import dowel
from dowel import logger
import time
import socket
from datetime import datetime


def restore_training(log_dir, exp_name, args, env_saved=True, env=None):
    # tabular_log_file = os.path.join(log_dir, 'progress_restored.{}.{}.csv'.
    #     format(str(time.time())[:10], socket.gethostname()))
    # text_log_file = os.path.join(log_dir, 'debug_restored.{}.{}.log'.
    #     format(str(time.time())[:10], socket.gethostname()))
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M")

    tabular_log_file = os.path.join(log_dir, f'progress_restored_{date_time}.csv')
    text_log_file = os.path.join(log_dir, f'debug_restored.{date_time}.log')
    logger.add_output(dowel.TextOutput(text_log_file))
    logger.add_output(dowel.CsvOutput(tabular_log_file))
    # logger.add_output(dowel.TensorBoardOutput(log_dir))
    logger.add_output(dowel.StdOutput())
    logger.push_prefix('[%s] ' % exp_name)

    ctxt = ExperimentContext(snapshot_dir=log_dir,
                             snapshot_mode='gap_and_last',
                             snapshot_gap=1)

    
    runner = LocalRunnerWrapper(
        ctxt,
        eval=args.eval_during_training,
        n_eval_episodes=args.n_eval_episodes,
        eval_greedy=args.eval_greedy,
        eval_epoch_freq=args.eval_epoch_freq,
        save_env=env_saved
    )
    saved = runner._snapshotter.load(log_dir, 'last')
    runner._setup_args = saved['setup_args']
    runner._train_args = saved['train_args']
    runner._stats = saved['stats']

    runner.auto_mode  = args.auto_mode
    print(f'[RESTORE TRAIN] ####### Training seed has been changed: {runner._setup_args.seed} --> {args.seed + 1} #######')
    runner._setup_args.seed = runner._setup_args.seed + 1

    set_seed(runner._setup_args.seed)
    algo = saved['algo']

    # Compatibility patch
    if not hasattr(algo, '_clip_grad_norm'):
        setattr(algo, '_clip_grad_norm', args.clip_grad_norm)

    if env_saved:
        env = saved['env']
        env.env.Rcom_th = env.env.Rcom_th.cpu()

    #! set device to Networks
    device = args.device
    set_nn_device_with(algo, device)
    #! set device to Networks----------------------------------------------------------------------

    runner.setup(env=env,
                 algo=algo,
                 sampler_cls=runner._setup_args.sampler_cls,
                 sampler_args=runner._setup_args.sampler_args,
                 hybrid_mode=args.hybrid,
                 devices=args.devices,
                 flag=args.flag,
                 )

    runner._train_args.start_epoch = runner._stats.total_epoch + 1
    runner._train_args.n_epochs = args.n_epochs # edit: Configure it to run up to n_epochs instead of executing n_epochs each time.
    
    print('\nRestored checkpoint from epoch #{}...'.format(runner._train_args.start_epoch))
    print('To be trained for additional {} epochs...'.format(args.n_epochs))
    print('Will be finished at epoch #{}...\n'.format(runner._train_args.n_epochs))

    return runner._algo.train(runner)