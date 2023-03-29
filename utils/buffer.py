import torch
import random
import numpy as np
from collections import deque
from typing import Deque, List, Dict

from utils.space import StateSpace, ActionSpace


# 经验回放
# 经验回放寄存器
class ReplayBuffer:
    def __init__(
        self, 
        state_space: StateSpace = None,
        action_space: ActionSpace = None,
        capacity: int = 1000
    ):
        self.action_space = action_space
        self.state_space = state_space
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)

    def store_transition(
        self, 
        state, 
        action, 
        reward, 
        next_state, 
        done: bool = False
    ):
        action_code = self.action_space.to_code(action)
        experience = (state, action_code, reward, next_state, done)
        while self.__len__() >= self.capacity:
            self.memory.popleft()
        self.memory.append(experience)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = list(zip(*random.sample(self.memory, batch_size)))
        states = torch.FloatTensor(np.array(states, dtype=float))
        actions = torch.LongTensor(np.array(actions, dtype=int))
        rewards = torch.FloatTensor(np.array(rewards, dtype=float))
        next_states = torch.FloatTensor(np.array(next_states, dtype=float))
        dones = torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones



class SingleReplayBuffer:
    def __init__(self, env, capacity):
        self.env = env
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        action = int(''.join(list(map(str, action))), self.env.num_actions)
        experience = (state, action, np.array([reward]), next_state, done)
        while self.__len__() >= self.capacity:
            self.memory.popleft()
        self.memory.append(experience)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = list(zip(*random.sample(self.memory, batch_size)))
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
    
    def test_store_transition(self, env):
        for _ in range(100):
            actions = np.random.choice(env.num_actions, env.num_drones)
            transition = env.step(actions)
            self.store_transition(*transition)
    
    def test_sample(self, env, batch_size):
        self.test_store_transition(env)
        return self.sample(batch_size)





# 并行经验回放寄存器组
class ParallelReplayBuffer:
    def __init__(self, num_envs, max_timesteps):
        self.max_timesteps = max_timesteps
        self.num_envs = num_envs