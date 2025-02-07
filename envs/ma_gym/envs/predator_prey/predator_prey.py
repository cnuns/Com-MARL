import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')

import copy
import logging

import gym
from PIL import ImageColor, Image, ImageDraw
from gym import spaces
from gym.utils import seeding

try:
    from ..utils.action_space import MultiAgentActionSpace
    from ..utils.observation_space import MultiAgentObservationSpace
    from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_sensing_outline, draw_circle_border

except:
    sys.path.append(current_file_path + '/../../../../')
    sys.path.append(current_file_path + '/../')
    from envs.utils.action_space import MultiAgentActionSpace
    from envs.utils.observation_space import MultiAgentObservationSpace
    from envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_sensing_outline, draw_circle_border
    
logger = logging.getLogger(__name__)

from custom_implement.env_communication import init_communication, print_env_params, update_communication_state, get_one_hot_vectors

import random
import numpy as np
import torch
import time

class PredatorPrey(gym.Env):
    """
    Predator-prey involves a grid world, in which multiple predators attempt to capture randomly moving prey.
    Agents have a 5 × 5 view and select one of five actions ∈ {Left, Right, Up, Down, Stop} at each time step.
    Prey move according to selecting a uniformly random action at each time step.

    We define the “catching” of a prey as when the prey is within the cardinal direction of at least one predator.
    Each agent’s observation includes its own coordinates, agent ID, and the coordinates of the prey relative
    to itself, if observed. The agents can separate roles even if the parameters of the neural networks are
    shared by agent ID.

    The terminating condition of this task is when all preys are caught by more than one predator.
    For every new episodes , preys are initialized into random locations. Also, preys never move by themself into
    predator's neighbourhood
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, **kwargs):
        params = kwargs['params']

        prey_move_probs = (0.175, 0.175, 0.175, 0.175, 0.3)
        full_observable = False
        self.epoch = None
        self.total_n_epi = 0
        self.epoch_n_epi = 0
        self._max_steps = self.Tmax = params.get("max_env_steps")
        self.L = params.get("n_gcn_layers")
        self.load = params.get("load")
        self.maps = params.get("grid_size")
        self._grid_shape = (self.maps, self.maps)
        self.n_agents = int(params.get("n_agents"))
        self.n_preys = int(params.get("n_preys"))
        self._prey_capture_reward = abs(params.get("capture_reward"))
        self._step_cost = -abs(params.get("step_cost"))
        self._moving_cost = -abs(params.get("rm"))
        self._penalty = -abs(params.get("penalty"))
        self.maps = params.get("grid_size")
        self.Rsen = params.get("Rsen")
        self._agent_view_mask = (2*self.Rsen+1, 2*self.Rsen+1)
        self.bound_return = self.n_preys * self._prey_capture_reward
        self.agent_condition = np.ones(self.n_agents)
        init_communication(self, params)
        
        if self.load == 2: self.capv = 1
        elif self.load == 3: self.capv = 3
        elif self.load == 4: self.capv = 4

        self.n_groups = params.get("n_groups")
        self.n_nodes = params.get("n_nodes")
        self.n_action = 5
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.n_action) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}
        self._prey_alive = None
        self.edges = self.__create_edges() 
        self._base_grid = self.__create_grid()
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
        self.viewer = None
        self.full_observable = full_observable
        mask_size = np.prod(self._agent_view_mask)
        
        self._obs_high = np.array([1.] * mask_size * 2 + [1., 1.] + [1.])
        self._obs_low = np.array([0.] * mask_size * 2 + [0., 0.] + [0.])
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self._obs_low = self._obs_low.astype(np.float32)
        self._obs_high = self._obs_high.astype(np.float32)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        param_print = params.get("env_param_print", 1)
        print_env_params(self, option=param_print)
        self._total_episode_reward = None

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_edges(self):
        edges = {}
        start, end = 0, self.maps-1
        for row in range(start, end+1, 1):
            for col in range(start, end+1, 1):
                pos = (row, col)
                if row in [start, end] and col in [start, end]:
                    edges[pos] = 2 # 'corner'
                elif row in [start, end] and col not in [start, end]: # on a line
                    edges[pos] = 3 # 'line'
                elif row not in [start, end] and col in [start, end]: # on a line
                    edges[pos] = 3 # 'line'
                else: # not on edge
                    pass

        num_edges = self.maps + self.maps + self.maps-2 + self.maps-2
        # color: upside -> downside -> left(-2) -> right(-2)

        if len(edges) != num_edges:
            raise Exception('edge number error')

        return edges
    
    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()
        self.agent_condition = np.ones(self.n_agents)

        for agent_i in range(self.n_agents):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    break

            self.__update_agent_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [random.randint(0, self._grid_shape[0] - 1), random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

        self.__draw_base_img()
    
    def get_neighbors(self, center_pos, target_obj='agent', fill=1):
        local_view = np.zeros(self._agent_view_mask)  # prey location in neighbour
        dy = dx = self.Rsen
        for row in range(max(0, center_pos[0] - dy), min(center_pos[0] + dy + 1, self._grid_shape[0])):
            for col in range(max(0, center_pos[1] - dx), min(center_pos[1] + dx + 1, self._grid_shape[1])):
                if PRE_IDS[target_obj] in self._full_obs[row][col]:
                    local_view[row - (center_pos[0] - dy), col - (center_pos[1] - dx)] = fill  # get relative position for the prey loc.

        return local_view
    
    def get_agent_obs(self):
        infos = None
        _obs = []
        for agent_i in range(self.n_agents):
            _agent_i_obs = []
            pos = self.agent_pos[agent_i]
            _agent_pos = self.get_neighbors(pos, 'agent')
            _agent_i_obs += _agent_pos.flatten().tolist()  # adding prey pos in observable area

            _prey_pos = self.get_neighbors(pos, 'prey')
            _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area

            _agent_i_obs += [pos[0] / self._grid_shape[0], pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time

            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]

        return _obs, infos

    def reset(self, epoch=-1):
        self.epoch = epoch
        self._total_capture_reward = 0
        self._total_step_cost = 0
        self._total_moving_cost = 0
        self._total_penalty = 0

        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        self.__init_full_obs()

        self.total_n_epi += 1
        self.epoch_n_epi += 1

        self._step_count = 0
        self._agent_dones = np.zeros(self.n_agents, dtype=bool)
        self._prey_alive = np.ones(self.n_preys, dtype=bool)
        
        update_communication_state(self)

        self.prev_action = [[0]*self.n_action for _ in range(self.n_agents)]
        _obs, infos = self.get_agent_obs()
        self.prev_infos = infos

        return _obs

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            if self.agent_condition[agent_i] != 0:
                self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey_view(prey_i)
            else:
                # print('pos not updated')
                pass
        else:
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_prey_view(self, prey_i):
        self._full_obs[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id
    
    def _neighbour_preys(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['prey'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['prey'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['prey'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['prey'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        prey_id = []
        for x, y in neighbours_xy:
            prey_id.append(int(self._full_obs[x][y].split(PRE_IDS['prey'])[1]) - 1)
        return _count, prey_id
    
    def _neighbour_objects(self, pos):
        # check if agent is in neighbour
        _count = {'0':0, 'A':0, 'P':0}
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            first_string = self._full_obs[pos[0] + 1][pos[1]][:1]
            _count[first_string] += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])

        if self.is_valid([pos[0] - 1, pos[1]]):
            first_string =  self._full_obs[pos[0] - 1][pos[1]][:1]
            _count[first_string] += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])

        if self.is_valid([pos[0], pos[1] + 1]):
            first_string =  self._full_obs[pos[0]][pos[1] + 1][:1]
            _count[first_string] += 1
            neighbours_xy.append([pos[0], pos[1] + 1])

        if self.is_valid([pos[0], pos[1] - 1]):
            first_string =  self._full_obs[pos[0]][pos[1] - 1][:1]
            _count[first_string] += 1
            neighbours_xy.append([pos[0], pos[1] - 1])

        agent_id = []
        for x, y in neighbours_xy:
            first_string = self._full_obs[x][y][:1]
            if first_string == PRE_IDS['agent']:
                agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id
    
    def _n_adjacent_grid(self, pos):
        return self.edges.get((pos[0],pos[1]), self.load) # Default = 4: if not on corner or line -> 4 direction

    def _apply_moving_cost(self, agents_action):
        if self._moving_cost != 0:
            agents_acts = np.where(agents_action != ACTION_INDICES['NOOP'], 1, 0)
            moving_costs = agents_acts * self._moving_cost
        else:
            moving_costs = np.zeros(self.n_agents)

        return moving_costs
    
    def prey_random_move(self, prey_i):
        prey_move = None
        if self._prey_alive[prey_i]:
            # 5 trails : we sample next move and check if prey (smart) doesn't go in neighbourhood of predator
            for _ in range(5):
                _move = np.random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
                if self._neighbour_agents(self.__next_pos(self.prey_pos[prey_i], _move))[0] == 0:
                    prey_move = _move
                    break
            prey_move = 4 if prey_move is None else prey_move  # default is no-op(4)

        self.__update_prey_pos(prey_i, prey_move)

    def reward_default(self, agents_action):
        rewards = 0
        capture_cnt = 0
        step_costs = self._step_cost
        penalty_cnt = 0
        moving_cnt = np.where(agents_action != ACTION_INDICES['NOOP'], 1, 0)
        prey_watching = [0]*self.n_agents

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.prey_pos[prey_i])

                for agent_i in n_i:
                    prey_watching[agent_i] = 1
                    
                if predator_neighbour_count >= 1:
                    if self.load <= predator_neighbour_count:
                        capture_cnt += 1
                        self._prey_alive[prey_i] = False
                    else:
                        penalty_cnt += 1
                        self._prey_alive[prey_i] = True
        
                self.prey_random_move(prey_i)
        
        rewards += (step_costs + self._prey_capture_reward*capture_cnt + self._moving_cost*sum(moving_cnt)/len(moving_cnt)) + self._penalty*penalty_cnt

        _obs, infos = self.get_agent_obs()
        
        self.prev_infos = infos
        
        reward_details = dict(
                            reward=rewards,
                            capture_cnt=(capture_cnt),
                            step_cnt=1,
                            move_cnt=np.mean(moving_cnt),
                            penalty_cnt=penalty_cnt,
                            variable=np.mean(prey_watching),
                            vars2=0
                            )
        
        return _obs, rewards, reward_details
    
    def reward_individual(self, agents_action):
        rewards = 0
        capture_cnt = 0
        step_costs = self._step_cost
        penalty_cnt = 0
        moving_cnt = np.where(agents_action != ACTION_INDICES['NOOP'], 1, 0)
        prey_watching = [0]*self.n_agents

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                neighbor_cnt, n_i = self._neighbour_objects(self.prey_pos[prey_i])
                predator_neighbour_count, prey_neighbour_count = neighbor_cnt['A'], neighbor_cnt['P']

                for agent_i in n_i:
                    prey_watching[agent_i] = 1

                if predator_neighbour_count >= 1:
                    n_neighbor_available_space = self._n_adjacent_grid(self.prey_pos[prey_i]) - prey_neighbour_count
                    if min(self.load, n_neighbor_available_space) <= predator_neighbour_count:
                        capture_cnt += 1
                        self._prey_alive[prey_i] = False

                    else:
                        penalty_cnt += 1
                        self._prey_alive[prey_i] = True

                self.prey_random_move(prey_i)
        
        rewards += (step_costs + self._prey_capture_reward*capture_cnt + self._moving_cost*sum(moving_cnt)/len(moving_cnt))
        
        reward_details = dict(
                            reward=rewards,
                            capture_cnt=(capture_cnt), 
                            step_cnt=1,
                            move_cnt=np.mean(moving_cnt),
                            penalty_cnt=penalty_cnt,
                            variable=np.mean(prey_watching),
                            vars2=0
                            )
        
        return rewards, reward_details

    def step(self, agents_action):
        self._step_count += 1

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.prev_action[agent_i] = ACTION_ONEHOT[action]
                self.__update_agent_pos(agent_i, action)

        if self.capv == 1:
            _obs, rewards, rewards_details = self.reward_default(agents_action)

        elif self.capv == 3 or self.capv == 4:
            (rewards, rewards_details) = self.reward_individual(agents_action)
            _obs, infos = self.get_agent_obs()

        update_communication_state(self)
        
        if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
            if True not in self._prey_alive:
                self.success = 1
            else:
                self.success = 0
            for i in range(self.n_agents):
                self._agent_dones[i] = True
        
        return _obs, (rewards,rewards_details), self._agent_dones, {'prey_alive': self._prey_alive}

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        # Capturing Range 
        for agent_i in range(self.n_agents):
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)



        # Agent Circle
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)
            
            # draw sensing range
            pos = self.agent_pos[agent_i]
            row, col = pos[0], pos[1]
            draw_sensing_outline(img, (row, col), Rsen=self.Rsen, cell_size=CELL_SIZE, fill=AGENT_SENSING_COLOR, width=2)

        # Prey Circle
        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def my_render(self, attention_weights=None, Rcom=None, cell_size=35):
        if attention_weights is not None:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = Image.fromarray(self.render(mode='rgb_array'))
            
            # Plot attention weights for agent 0
            start_col, start_row = self.agent_pos[0]
            start_x, start_y = (start_row + 0.5) * cell_size, (start_col + 0.5) * cell_size
            for i in range(self.n_agents):
                if i == 0:
                    # Plot Rcom Range
                    if Rcom == None:
                        Rcom = 0.3
                    
                    draw_circle_border(img, self.agent_pos[i], cell_size=cell_size, fill='green', radius=Rcom, outline='green')
                else:
                    if attention_weights[i] == 0:
                        fill = None
                    else:
                        fill = 'green'

                        end_col, end_row = self.agent_pos[i]
                        end_x, end_y = (end_row + 0.5) * cell_size, (end_col + 0.5) * cell_size
                        ImageDraw.Draw(img).line(((start_x, start_y), (end_x, end_y)), 
                            fill=fill, width=int(20 * attention_weights[i]))

            img = np.asarray(img)
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            self.render(mode='human')


    def seed(self, n):
        self.np_random, seed1 = seeding.np_random(n)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
    

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_SENSING_COLOR = (47,157,39) # ImageColor.getcolor('green', mode='RGB')

GROUP_COLORS = [(0,0,255), # blue
                (255,0,0), # red
                (255,0,255), # magenta
                (0,255,255), # cyan
                (255,165,0), # orange
                (128,0,128), # purple
                (255,192,203), # pink
                (64,224,208),
                ] #

AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

ACTION_INDICES = {
    'DOWN': 0,
    'LEFT': 1,
    'UP': 2,
    'RIGHT': 3,
    'NOOP': 4,
}

ACTION_LIST = [0, 1, 2, 3, 4]

ACTION_ONEHOT = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]


PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}

def fix_randomness(seed):
    import torch
    #* randomness set
    deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    params = {
        'load': 2,
        'max_env_steps': 200,
        'InputType': 'default',
        'OutputType': None,
        'RewDesign': None,
        'SpawnType': 'random',

        'capture_reward': 10,
        'step_cost': -0.1,
        'rm': 0,
        'penalty': 0,
        'grid_size': 20,
        'Rsen': 1,
        'env_param_print': 1,
        'n_groups': 1,
        'n_nodes': 1,
        'n_agents':16,
        'n_preys': 16,

        'n_gcn_layers': 2,
        'curriculum_learning': 0,
        
        'InputChannelInfo': 1,
        'channelType': 'IID',
        'unit_time': None,
        'LinkReliability': None,
        'loss_apply': 1,
        'transmit': None,
        
        'isbad_ratio': None,
        'isburst_length': None,
        'isNswitches': None,
        'Pgb': None,
        'Pbg': None,

        'isRcomSame': True,
        'isEuclidean': True,
        'calc_diameter': None,

        'mode': 'train',
        'trRcom': 9,
        'teRcom': 9,
        'trplPmin': None,
        'trplPmax': None,
        'trpltype': 'iid',
        'trpl':0,
        'trpfNmin': None,
        'trpfNmax': None,
        'trpftype': 'iid',
        'trpf':0,
    }

    fix_randomness(1)
    
    from torch.distributions import Categorical
    class simpleActNet(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()

            self.input_size = input_size
            self.output_size = output_size
            self.acts = list(ACTION_MEANING.keys())
            self.linear = torch.nn.Linear(input_size, output_size)
            
        def forward(self, x):
            probs = self.linear(x) #! this is for input/output size check
            # actions = selected_elements = np.random.choice(self.acts, size=params['n_agents'])
            dists_n = Categorical(logits=probs)
            actions_n = dists_n.sample().numpy()
            return actions_n
    
    env = PredatorPrey(params=params)
    Net = simpleActNet(input_size=env.observation_space[0].shape[0], output_size=env.n_action)
    Net.eval()

    obs = env.reset()
    full_conn = np.ones(env.n_agents)
    attention_weights = full_conn * (env.dist_adj) * (env.channels[-1])
    attention_weights = attention_weights / (attention_weights.sum(axis=-1) + 1e-7)

    env.my_render(attention_weights[0], env.Rcom_th.item())
    time.sleep(1.0)
    
    for i in range(1000):
        actions = Net(torch.Tensor(obs))
        obs, (rewards,reward_details), dones, _ = env.step(actions)
        attention_weights = np.ones(env.n_agents) * (env.dist_adj) * (env.channels[-1])
        attention_weights = attention_weights / (attention_weights.sum(axis=-1) + 1e-7)

        if all(dones):
            env.my_render(attention_weights[0], env.Rcom_th.item())
            time.sleep(1.0)

            obs = env.reset()
            env.my_render(attention_weights[0], env.Rcom_th.item())
            time.sleep(1.0)
        else:
            env.my_render(attention_weights[0], env.Rcom_th.item())
            
        