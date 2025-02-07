import random
import numpy as np
import torch
import time
from collections import Counter
from testing import VECTORS
from custom_implement.randomness import fix_randomness

def eval_model(env, policy, itr, n_eval_episodes=100, max_env_steps=200,
               eval_greedy=True, render=False, inspect_steps=False, seed=1, flag=None):
    env.eval_n_epi = 0
    fix_randomness(seed)

    # Eval stats:
    traj_len = []
    
    #! ######################
    eval_rewards = []
    episode_data = []
    epi_success = []
    epi_rewards = {vec:[] for vec in VECTORS}
    link_losses = []
    #! ######################

    for i_eps in range(n_eval_episodes):
        if flag[0]:
            return None, None, None, None
        print(f'Eval episode: {i_eps}/{n_eval_episodes}', end=' ')
        obses = env.reset()
        link_losses.append(env.pl)
        # print('pos:', env.agent_pos)
        env.success = 0

        policy.reset([True])
        policy.step = 0
        eval_rewards.append(0)
        #! ######################
        step_success = []
        step_data = {vec:[] for vec in VECTORS}
        #! ######################
        
        for i_step in range(max_env_steps):
            if hasattr(policy, 'comm'):
                actions, agent_infos = policy.get_actions(obses, env.get_avail_actions(),
                                                                torch.Tensor(env.dist_adj),
                                                                torch.Tensor(env.channels),
                                                                greedy=eval_greedy)
                attention_weights_0 = np.array(agent_infos['attention_weights'])
                attention_weights_0 = attention_weights_0[0]
            else:
                actions, agent_infos = policy.get_actions(obses,
                                                          env.get_avail_actions(),
                                                          greedy=eval_greedy)
                
            actions = actions if len(actions.shape) == 1 else actions[0]

    
            if bool(render):
                env.my_render(attention_weights=None)
                if bool(inspect_steps):
                    input('Step {}, press Enter to continue...'.format(i_step))
                else:
                    time.sleep(0.3)
            
            obses, (reward, details), agent_dones, _ = env.step(actions)
            eval_rewards[-1] += reward
            
            step_success.append(env.success)
            for vec in VECTORS:
                if vec == 'nodeDeg': step_data['nodeDeg'].append(env.ave_deg)
                else: step_data[vec].append(details[vec])
                
            if agent_dones or i_step == max_env_steps - 1:
                if i_step < max_env_steps - 1:
                    traj_len.append(i_step + 1)
                    success = 1
                else:
                    success = 0
                    #print('eps {} captured all preys in {} steps'.format(i_eps + 1, i_step + 1))
                    
                epi_success.append(env.success)
                for vec in VECTORS:
                    if vec == 'nodeDeg':
                        epi_rewards[vec].append(np.mean(step_data[vec]))
                    else:
                        epi_rewards[vec].append(np.sum(step_data[vec]))
                #! ######################
                
                rew = np.sum(step_data['reward'])      
                print(f' || Reward: {rew:.2f} || Success: {success} || Ploss: {env.pl}')
                break
                
            
            policy.step += 1

        episode_data.append((step_success, step_data))
    print(f'### Average Reward = {(sum(eval_rewards) / len(eval_rewards)) / env.bound_return}')

    c = Counter(link_losses)
    cnt = list(c.values())
    print(f'### loss ratio = {list(c.keys())} = {np.array(cnt)/sum(cnt)}')
    env.close()

    return episode_data, epi_success, epi_rewards, env.bound_return


def eval_simple(args, env, algo):
    # Eval stats:
    distance_vs_weight = {}
    traj_len = []
    start = time.time()
    for i_eps in range(args.n_eval_episodes):
        print(f'Eval episode: {i_eps + 1}/{args.n_eval_episodes}', end=' || ')
        
        obses = env.reset()
        algo.policy.reset([True])
        for i_step in range(args.max_env_steps):
            if hasattr(algo.policy, 'comm'):
                actions, agent_infos = algo.policy.get_actions(obses,
                                                            env.get_avail_actions(),
                                                            torch.Tensor(env.dist_adj),
                                                            torch.Tensor(env.channels),
                                                            greedy=args.eval_greedy)
            else:
                actions, agent_infos = algo.policy.get_actions(obses,
                                                            env.get_avail_actions(),                                                                       
                                                            greedy=args.eval_greedy)
                
            if bool(args.render):
                attention_weights_0 = None
                env.my_render(attention_weights=attention_weights_0)
                
                if i_step == 0:
                    time.sleep(1.3)
                    
                if bool(args.inspect_steps):
                    input('Step {}, press Enter to continue...'.format(i_step))
                else:
                    time.sleep(0.43)
            
            obses, _, agent_dones, _ = env.step(actions)

            if agent_dones:
                if i_step < args.max_env_steps - 1:
                    traj_len.append(i_step + 1)
                    print(f'eps {i_eps+1} captured all preys in {i_step + 1} steps', end=' ||  ')
                    
                env.my_render(attention_weights=attention_weights_0)
                time.sleep(1.3)
                
                print()
                break
            
                
    env.close()
    print('Average trajectory length = {}'.format(np.mean(traj_len)))
    test_time = time.time() - start
    print(f'test_time: {test_time}')
