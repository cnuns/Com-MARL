import random
import numpy as np
import torch
import time
from randomness import fix_randomness

def eval_model(env, policy, itr,
          n_eval_episodes=100, max_env_steps=200, eval_greedy=True, render=False,
          inspect_steps=False, seed=1):

    fix_randomness(seed)
    
    # Eval stats:
    data = {'iter':[], 'avgTotalRew':[], 'captureRew':[], 'stepCost':[], 'movingCost':[], 'penalty':[], 'nodeDeg':[]}
    traj_len = []

    for i_eps in range(n_eval_episodes):
        print(f'Eval episode: {i_eps}/{n_eval_episodes}', end=' ')
        obses = env.reset()
        policy.reset([True])
        policy.step = 0

        rew_dict = {'avgTotalRew':0, 'captureRew':0, 'stepCost':0, 'movingCost':0, 'penalty':0, 'nodeDeg':0}
        
        for i_step in range(max_env_steps):
            actions, agent_infos = policy.get_actions(obses, env.get_avail_actions(),
                                                            torch.FloatTensor(env.dist_adj),
                                                            torch.FloatTensor(env.channels),
                                                            torch.FloatTensor(env.delays),
                                                            torch.FloatTensor([env.delay_th]),
                                                            greedy=eval_greedy)
            
            attention_weights_0 = np.array(agent_infos['attention_weights'])

            if bool(render):
                attention_weights_0 = attention_weights_0[0]
                env.my_render(attention_weights=None)
                if bool(inspect_steps):
                    input('Step {}, press Enter to continue...'.format(i_step))
                else:
                    time.sleep(0.3)
            
            obses, (reward, details), agent_dones, _ = env.step(actions)
            rew_dict['avgTotalRew'] += reward
            rew_dict['captureRew'] += details['capture_reward']
            rew_dict['stepCost'] += details['step_cost']
            rew_dict['movingCost'] += details['moving_cost']
            rew_dict['penalty'] += details['penalty']
            rew_dict['nodeDeg'] += env.ave_deg

            if agent_dones or i_step == max_env_steps - 1:
                if i_step < max_env_steps - 1:
                    traj_len.append(i_step + 1)
                    #print('eps {} captured all preys in {} steps'.format(i_eps + 1, i_step + 1))
                data['iter'].append(itr)
                data['avgTotalRew'].append(rew_dict['avgTotalRew'])
                data['captureRew'].append(rew_dict['captureRew'])
                data['stepCost'].append(rew_dict['stepCost'])
                data['movingCost'].append(rew_dict['movingCost'])
                data['penalty'].append(rew_dict['penalty']) 
                data['nodeDeg'].append(rew_dict['nodeDeg'] / i_step)

                rew = rew_dict['avgTotalRew']
                print(f' || Reward: {rew:.2f}')
                break

            policy.step += 1

    env.close()
    #print('Average trajectory length = {}'.format(np.mean(traj_len)))
    return data