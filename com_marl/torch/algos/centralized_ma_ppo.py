import sys
import os

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../../')

from custom_implement.monitoring_gpu_usage import get_gpu_memory

import collections
import copy

from dowel import tabular, logger
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance, TrajectoryBatch
from garage.misc import tensor_utils
from garage.torch.algos import (_Default, compute_advantages, filter_valids,
                                make_optimizer, pad_to_last)
from garage.torch.utils import flatten_batch
from garage.np.baselines import LinearFeatureBaseline

from com_marl.np.algos import MABatchPolopt
from com_marl.torch.algos.utils import pad_one_to_last
from .my_optimizer.adam import Adam

import time
import random
import gc
import os
from datetime import datetime

def get_gpu_alloc(device=0):
    device = torch.device(f"cuda:{device}")
    mem = torch.cuda.memory_allocated(device=device)
    return f'{(mem)/(1024**3):.2f}'

class CentralizedMAPPO(MABatchPolopt):
    """Centralized Multi-agent Vanilla Policy Gradient (REINFORCE).

    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.
        optimizer (Union[type, tuple[type, dict]]): Type of optimizer.
            This can be an optimizer type such as `torch.optim.Adam` or a
            tuple of type and dictionary, where dictionary contains arguments
            to initialize the optimizer e.g. `(torch.optim.Adam, {'lr' = 1e-3})`
        policy_lr (float): Learning rate for policy parameters.
        max_path_length (int): Maximum length of a single rollout.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(
            self,
            env_spec,
            policy,
            baseline,
            optimizer=Adam,#torch.optim.Adam,
            baseline_optimizer=Adam,#torch.optim.Adam,
            optimization_n_minibatches=1,
            optimization_mini_epochs=1,
            policy_lr=_Default(3e-4),
            lr_clip_range=2e-1,
            max_path_length=500,
            num_train_per_epoch=1,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            policy_ent_coeff=0.0,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            entropy_method='no_entropy',
            clip_grad_norm=None,
            device='cpu',
    ):
        
        self.device = device
        self.gpu_mem_history = []
        self.gpu_mem_max = (-1,-1)


        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._lr_clip_range = 0.1
        self._eps = 1e-8

        self.kl_flag = 0

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)

        self._optimizer = make_optimizer(optimizer,
                                         policy,
                                         lr=policy_lr,
                                         eps=_Default(1e-5),
                                         device=self.device
                                         )

        if not isinstance(baseline, LinearFeatureBaseline):
            self._baseline_optimizer = make_optimizer(baseline_optimizer,
                                                      baseline,
                                                      lr=policy_lr,
                                                      eps=_Default(1e-5),
                                                      device=self.device,
                                                      )

        self._optimization_n_minibatches = optimization_n_minibatches
        self._optimization_mini_epochs = optimization_mini_epochs

        self._clip_grad_norm = clip_grad_norm

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         discount=discount,
                         max_path_length=max_path_length,
                         n_samples=num_train_per_epoch,
                         )

        self._old_policy = copy.deepcopy(self.policy)

    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    def train_once(self, runner):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            dict: Processed sample data, with key
                * average_return: (float)

        """

        itr, paths = runner.step_itr,runner.step_path
        self.gpu_mem_history = []
        logger.log('Processing samples...')
        obs, avail_actions, actions, rewards, valids, baselines, returns, dist_adjs, channels= self.process_samples(itr, paths)
        MEM = get_gpu_alloc()
        self.gpu_mem_history.append(float(MEM))
        logger.log('GPU MEM Process Sample: {}'.format(MEM,))
        print('processed obs.shape =', obs.shape)

        with torch.no_grad():

            loss_before = self._compute_loss(itr, obs, avail_actions, actions, 
                                             rewards, valids, baselines, dist_adjs, channels)
            torch.cuda.empty_cache() #!
            kl_before = self._compute_kl_constraint(obs, avail_actions, dist_adjs, channels, actions)
            torch.cuda.empty_cache() #!
        self._old_policy.load_state_dict(self.policy.state_dict())

        # Start train with path-shuffling
        grad_norm = []
        step_size = int(np.ceil(len(rewards) / self._optimization_n_minibatches))
        shuffled_ids = np.random.permutation(len(rewards))
        print('MultiAgentNumTrajs =', len(rewards))

        t = time.time()
        for mini_epoch in range(self._optimization_mini_epochs):
            for start in range(0, len(rewards), step_size):
                ids = shuffled_ids[start : min(start + step_size, len(rewards))]
                print('Mini epoch: {} | Optimizing policy using traj {} to traj {}'.
                    format(mini_epoch, start, min(start + step_size, len(rewards)))
                )

                if mini_epoch == self._optimization_mini_epochs - 1:
                    MEM = get_gpu_alloc()
                    self.gpu_mem_history.append(float(MEM))
                    logger.log('GPU MEM Compute Loss: {}'.format(MEM,))

                loss = self._compute_loss(itr, obs[ids], avail_actions[ids], 
                                          actions[ids], rewards[ids], 
                                          valids[ids], baselines[ids], dist_adjs[ids], channels[ids])

                if not isinstance(self.baseline, LinearFeatureBaseline):
                    if self.baseline.name in ['base_critic']:
                        baseline_loss = self.baseline.compute_loss(obs[ids], returns[ids], dist_adjs[ids], channels[ids])
                    else:
                        baseline_loss = self.baseline.compute_loss(obs[ids], returns[ids])

                    # print(comm_powers[ids])
                    self._baseline_optimizer.zero_grad()

                    if mini_epoch == self._optimization_mini_epochs - 1:
                        MEM = get_gpu_alloc()
                        self.gpu_mem_history.append(float(MEM))
                        logger.log('GPU MEM Compute Loss: {}'.format(MEM,))

                    baseline_loss.backward()
                self._optimizer.zero_grad()

                if mini_epoch == self._optimization_mini_epochs - 1:
                    MEM = get_gpu_alloc()
                    self.gpu_mem_history.append(float(MEM))
                    logger.log('GPU MEM Compute Loss: {}'.format(MEM,))

                loss.backward()
                
                if self._clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                                    self._clip_grad_norm)
                grad_norm.append(self.policy.grad_norm())
                
                self._optimize(itr, obs[ids], avail_actions[ids], actions[ids], 
                               rewards[ids], valids[ids], baselines[ids], returns[ids])
            logger.log('Mini epoch: {} | Loss: {}'.format(mini_epoch, loss))
            if not isinstance(self.baseline, LinearFeatureBaseline):
                logger.log('Mini epoch: {} | BaselineLoss: {}'.format(mini_epoch, 
                                                                      baseline_loss))
                
        torch.cuda.empty_cache()#!

        epoch_time = time.time() - t
        grad_norm = np.mean(grad_norm)
        # End train

        with torch.no_grad():
            loss_after = self._compute_loss(itr, obs, avail_actions, actions, 
                                            rewards, valids, baselines, dist_adjs, channels)
            torch.cuda.empty_cache() #!
            kl = self._compute_kl_constraint(obs, avail_actions,dist_adjs, channels, actions)
            policy_entropy = self._compute_policy_entropy(obs, avail_actions, dist_adjs, channels, actions)
            torch.cuda.empty_cache() #!

        MEM = get_gpu_alloc()
        self.gpu_mem_history.append(float(MEM))
        logger.log('GPU MEM After BProp: {}'.format(MEM,))

        if isinstance(self.baseline, LinearFeatureBaseline):
            logger.log('Fitting baseline...')
            self.baseline.fit(paths)

        # logging ##############################################################
        # log_performance customization block
        n_agents = actions.shape[-1]
        returns = []
        successes = []
        undiscounted_returns = []

        #!----
        sum_capture_rewards = []
        sum_step_costs = []
        sum_moving_costs = []
        sum_penalties = []
        sum_variables = []
        sum_vars2 = []
        #!----
        #*-----------
        ave_degs = []
        diameters = []
        ave_throughputs = []
        #*-----------

        for i_path in range(len(paths)):
            successes.append(paths[i_path]['success'])
            path_rewards = np.asarray(paths[i_path]['rewards'])
            returns.append(paths[i_path]['returns'])
            undiscounted_returns.append(np.sum(path_rewards))
            ave_deg = np.asarray(paths[i_path]['ave_degs'])
            diameter = np.asarray(paths[i_path]['diameters'])
            ave_throughput = np.asarray(paths[i_path]['ave_trputs'])
            ave_degs.append(np.mean(ave_deg))
            diameters.append(np.mean(diameter))
            ave_throughputs.append(np.mean(ave_throughput))

            capture_rewards = []
            step_costs = []
            moving_costs = []
            penalties = []
            variables = []
            vars2 = []
            
            for rew_dicts in paths[i_path]['rewards_details']:
                capture_rewards.append(rew_dicts['capture_cnt'])
                step_costs.append(rew_dicts['step_cnt'])
                moving_costs.append(rew_dicts['move_cnt'])
                penalties.append(rew_dicts['penalty_cnt'])
                variables.append(rew_dicts['variable'])
                vars2.append(rew_dicts['vars2'])

            sum_capture_rewards.append(np.sum(np.asarray(capture_rewards)))
            sum_step_costs.append(np.sum(np.asarray(step_costs)))
            sum_moving_costs.append(np.sum(np.asarray(moving_costs)))
            sum_penalties.append(np.sum(np.asarray(penalties)))
            sum_variables.append(np.sum(np.asarray(variables)))
            sum_vars2.append(np.sum(np.asarray(vars2)))

        average_returns = undiscounted_returns
        average_discounted_return = np.mean([r[0] for r in returns])

        tabular.record('Iteration', itr)
        tabular.record('NumTrajs', len(paths) * self.policy._n_agents)
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('SuccessRate', np.mean(successes))

        tabular.record('AverageCaptureCount', np.mean(sum_capture_rewards))
        tabular.record('AverageStepCount', np.mean(sum_step_costs))
        tabular.record('AverageMovingCount', np.mean(sum_moving_costs))
        tabular.record('AveragePenaltyCount', np.mean(sum_penalties))
        tabular.record('AverageVariable', np.mean(sum_variables))
        tabular.record('AverageVar2', np.mean(sum_vars2))

        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('LossBefore', loss.item())
        tabular.record('LossAfter', loss_after.item())
        tabular.record('dLoss', loss.item() - loss_after.item())
        tabular.record('KLBefore', kl_before.item())
        tabular.record('KL', kl.item())
        tabular.record('Entropy', policy_entropy.mean().item())
        tabular.record('GradNorm', grad_norm)
        tabular.record('EpochTime', epoch_time)

        tabular.record('AveDegree', np.mean(ave_degs))
        tabular.record('Diameter', np.mean(diameters))
        tabular.record('AveTroughput', np.mean(ave_throughputs))

        MEM = get_gpu_alloc()
        self.gpu_mem_history.append(float(MEM))
        logger.log('GPU MEM Before Clear: {}'.format(MEM,))

        torch.cuda.empty_cache()
        gc.collect()

        MEM = get_gpu_alloc()
        self.gpu_mem_history.append(float(MEM))
        logger.log('GPU MEM After Clear: {}'.format(MEM,))
        new_max = max(self.gpu_mem_history) 
        tabular.record('GPUMemoryMax', new_max)
        
        self.gpu_mem_max = (itr, new_max) if self.gpu_mem_max[1] < new_max else (self.gpu_mem_max[0], self.gpu_mem_max[1])
        return np.mean(average_returns)

    def _compute_loss(self, itr, obs, avail_actions, actions, rewards, valids, 
                      baselines, dist_adjs, channels):
        """Compute mean value of loss.

        Args:
            itr (int): Iteration number.
            obs (torch.Tensor): Observation from the environment.
            actions (torch.Tensor): Predicted action.
            rewards (torch.Tensor): Feedback from the environment.
            valids (list[int]): Array of length of the valid values.
            baselines (torch.Tensor): Value function estimation at each step.

        Returns:
            torch.Tensor: Calculated mean value of loss

        """
        # print(baselines.shape)
        
        del itr

        if self.policy.recurrent:
            policy_entropies = self._compute_policy_entropy(obs, avail_actions, dist_adjs, channels, actions)
        else:
            policy_entropies = self._compute_policy_entropy(obs, avail_actions, dist_adjs, channels)

        if self._maximum_entropy:
            rewards += self._policy_ent_coeff * policy_entropies

        advantages = compute_advantages(self.discount, self._gae_lambda,
                                        self.temp_max_path_length, baselines,
                                        rewards, self.device)

        if self._center_adv:
            if len(advantages.shape) == 1:
                advantages = advantages.reshape(1, -1)
            means, variances = list(zip(*[(valid_adv.mean(), valid_adv.var(unbiased=False)) for valid_adv in filter_valids(advantages, valids)]))
            advantages = F.batch_norm(advantages.t(), torch.Tensor(means).to(self.device), torch.Tensor(variances).to(self.device), eps=self._eps).t()

        if self._positive_adv:
            advantages -= advantages.min()

        objective = self._compute_objective(advantages, valids, obs, avail_actions, actions, rewards, dist_adjs, channels)


        if self._entropy_regularzied:
            objective += self._policy_ent_coeff * policy_entropies

        valid_objectives = filter_valids(objective, valids)
        return -torch.cat(valid_objectives).mean()

    def _compute_kl_constraint(self, obs, avail_actions, dist_adjs, channels, actions=None):
        """Compute KL divergence.

        Compute the KL divergence between the old policy distribution and
        current policy distribution.

        Args:
            obs (torch.Tensor): Observation from the environment.

        Returns:
            torch.Tensor: Calculated mean KL divergence.

        """
        if self.policy.recurrent:
            with torch.no_grad():
                if hasattr(self.policy, 'comm'):
                    old_dist, _ = self._old_policy.forward(obs_n=obs, 
                                                           avail_actions_n=avail_actions, 
                                                           dist_adj=dist_adjs, 
                                                           channels=channels, 
                                                           actions_n=actions,
                                                           )

                else:
                    old_dist = self._old_policy.forward(obs, avail_actions, actions)
    
            if hasattr(self.policy, 'comm'):
                new_dist, _ = self.policy.forward(obs_n=obs, 
                                                  avail_actions_n=avail_actions, 
                                                  dist_adj=dist_adjs, 
                                                  channels=channels,                                                  
                                                  actions_n=actions,
                                                  )
            else:
                new_dist = self.policy.forward(obs, avail_actions, actions)
        
        else:

            flat_obs = flatten_batch(obs)
            flat_avail_actions = flatten_batch(avail_actions)
            flat_dist_adjs = flatten_batch(dist_adjs)
            flat_channels = flatten_batch(channels)

            with torch.no_grad():
                if hasattr(self.policy, 'comm'):
                    old_dist, _ = self._old_policy.forward(flat_obs, flat_avail_actions, flat_dist_adjs, flat_channels, get_actions=False)
                else:
                    old_dist = self._old_policy.forward(flat_obs, flat_avail_actions, get_actions=False)
    
            if hasattr(self.policy, 'comm'):
                new_dist, _ = self.policy.forward(flat_obs, flat_avail_actions, flat_dist_adjs, flat_channels, get_actions=False)
            else:
                new_dist = self.policy.forward(flat_obs, flat_avail_actions, get_actions=False)
    
        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)
    
        return kl_constraint.mean()

    def _compute_policy_entropy(self, obs, avail_actions, dist_adj, channels, actions=None):
        """Compute entropy value of probability distribution.

        Args:
            obs (torch.Tensor): Observation from the environment.

        Returns:
            torch.Tensor: Calculated entropy values given observation

        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                if self.policy.recurrent:
                    policy_entropy = self.policy.entropy(obs, avail_actions, actions)
                else:
                    policy_entropy = self.policy.entropy(obs, avail_actions)
        else:
            if self.policy.recurrent:
                if hasattr(self.policy, 'comm'):
                    policy_entropy = self.policy.entropy(observations=obs,
                                                        avail_actions=avail_actions,
                                                        dist_adj=dist_adj,
                                                        channels=channels,
                                                        actions=actions,
                    )

                else:
                    policy_entropy = self.policy.entropy(obs, avail_actions)
                    
            else:
                if hasattr(self.policy, 'comm'):
                    policy_entropy = self.policy.entropy(obs, avail_actions, dist_adj, channels)
                else:
                    policy_entropy = self.policy.entropy(obs, avail_actions)

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, valids, obs, avail_actions, 
                           actions, rewards, dist_adj, channels):
        """Compute objective value.

        Args:
            advantages (torch.Tensor): Expected rewards over the actions.
            valids (list[int]): Array of length of the valid values.
            obs (torch.Tensor): Observation from the environment.
            actions (torch.Tensor): Predicted action.
            rewards (torch.Tensor): Feedback from the environment.

        Returns:
            torch.Tensor: Calculated objective values

        """
        # Compute constraint
        with torch.no_grad():
            if hasattr(self._old_policy, 'comm'):
                old_ll = self._old_policy.log_likelihood(observations=obs,
                                                         avail_actions=avail_actions,
                                                         dist_adj=dist_adj,
                                                         channels=channels,
                                                         actions=actions,
                )
            else:
                old_ll = self._old_policy.log_likelihood(obs, avail_actions, actions)

        if hasattr(self._old_policy, 'comm'):
            new_ll = self.policy.log_likelihood(observations=obs,
                                                avail_actions=avail_actions,
                                                dist_adj=dist_adj,
                                                channels=channels,
                                                actions=actions,)
        else:
            new_ll = self.policy.log_likelihood(obs, avail_actions, actions)

        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        # Clipping the constraint
        likelihood_ratio_clip = torch.clamp(likelihood_ratio,
                                            min=1 - self._lr_clip_range,
                                            max=1 + self._lr_clip_range)

        # Calculate surrotate clip
        surrogate_clip = likelihood_ratio_clip * advantages

        return torch.min(surrogate, surrogate_clip)

    def _get_baselines(self, path):
        """Get baseline values of the path.

        Args:
            path (dict): collected path experienced by the agent

        Returns:
            torch.Tensor: A 2D vector of calculated baseline with shape(T),
                where T is the path length experienced by the agent.

        """
        if hasattr(self.baseline, 'predict_n'):
            return torch.Tensor(self.baseline.predict_n(path))
        return torch.Tensor(self.baseline.predict(path))

    def _optimize(self, itr, obs, avail_actions, actions, rewards, valids, baselines, returns):
        del itr, valids, obs, avail_actions, actions, rewards, baselines, returns
        self._optimizer.step()
        if not isinstance(self.baseline, LinearFeatureBaseline):
            self._baseline_optimizer.step()

    def process_samples(self, itr, paths):
        """Process sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            tuple:
                * obs (torch.Tensor): The observations of the environment.
                * actions (torch.Tensor): The actions fed to the environment.
                * rewards (torch.Tensor): The acquired rewards.
                * valids (list[int]): Numbers of valid steps in each paths.
                * baselines (torch.Tensor): Value function estimation
                    at each step.

        """

        self.temp_max_path_length = 0
        for path in paths:
            if len(path['rewards']) > self.temp_max_path_length:
                self.temp_max_path_length = len(path['rewards'])

            if 'returns' not in path:
                path['returns'] = tensor_utils.discount_cumsum(path['rewards'], self.discount)
        
        # 100-cycle calc time: 18 sec.
        #* Padding(discounted cumulative return)
        returns = torch.stack([pad_to_last(tensor_utils.discount_cumsum(path['rewards'], self.discount).copy(), total_length=self.temp_max_path_length) for path in paths]).to(self.device)
        valids = torch.Tensor([len(path['actions']) for path in paths]).int().to(self.device)
        obs = torch.stack([pad_to_last(path['observations'],total_length=self.temp_max_path_length,axis=0) for path in paths]).to(self.device)
        avail_actions = torch.stack([pad_one_to_last(path['avail_actions'],total_length=self.temp_max_path_length,axis=0) for path in paths]).to(self.device) # Cannot pad all zero since prob sum cannot be zero
        actions = torch.stack([pad_to_last(path['actions'],total_length=self.temp_max_path_length,axis=0) for path in paths]).to(self.device)
        rewards = torch.stack([pad_to_last(path['rewards'], total_length=self.temp_max_path_length) for path in paths]).to(self.device)

        dist_adjs = torch.stack([pad_one_to_last(path['dist_adjs'],total_length=self.temp_max_path_length,axis=0) for path in paths]).to(self.device)
        channels = torch.stack([pad_one_to_last(path['channels'],total_length=self.temp_max_path_length,axis=0) for path in paths]).to(self.device)

        if isinstance(self.baseline, LinearFeatureBaseline):
            baselines = torch.stack([pad_to_last(self._get_baselines(path),total_length=self.temp_max_path_length) for path in paths]).to(self.device)
        else:
            with torch.no_grad():
                if self.baseline.name in ['base_critic']:
                    baselines = self.baseline.forward(obs, avail_actions, dist_adjs, channels)
                else:
                    baselines = self.baseline.forward(obs) # (n_epi, max_step)
        
        return obs, avail_actions, actions, rewards, valids, baselines, returns, dist_adjs, channels
    # baselines = V_pi(s)
    # returns = Q_pi(s,a)