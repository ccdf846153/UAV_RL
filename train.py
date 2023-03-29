import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List
import logging
import time

from utils.env import UAVEnvironment
from utils.buffer import ReplayBuffer
from utils.env_args import *
from utils.space import ActionSpace, StateSpace
from modules.Net import Net
from modules.RL import *


logging.basicConfig(
    filename = f"./train_result/train_log__{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S',
    level = logging.INFO
)


class UAVAgent:
    def __init__(
        self, 
        env_list: List[UAVEnvironment],
        algorithm: str = 'DQN',
        interact_distrib: List[float] = [1],
        sample_distrib: List[float] = [1]
    ):
        self.env_list = env_list
        self.state_space = self.env_list[0].state_space
        self.action_space = self.env_list[0].action_space
        self.model: ReinforcementLearning = eval(algorithm)(
            state_space = self.state_space,
            action_space = self.action_space,
            eval_net = Net(
                len(self.state_space),
                len(self.action_space)
            ),
            target_net = Net(
                len(self.state_space),
                len(self.action_space)
            ),
            gamma = 0.9,
            loss_function = nn.MSELoss(),
            optim_info = {
                'net': 'eval_net',
                'type': 'Adam',
                'lr': LR
            },
            buffer_info = {
                'type': 'ReplayBuffer',
                'capacity': MEMORY_CAPACITY,
                'num': len(self.env_list)
            },
            batch_size = BATCH_SIZE,
            target_replace_step = TARGET_REPLACE_ITER / len(self.env_list)
        )
        
        self.interact_distrib = interact_distrib
        self.sample_distrib = sample_distrib
        self.num_envs = len(self.env_list)
        self.algorithm = algorithm

    def train(self, dict_input=None):
        # * 以下代码，在运行之前请保证 self.env_list[0] 对应现实中的真实环境
        reward_list = []
        rate_list = []

        for episode in range(10000):                                                    # 400个episode循环
            if dict_input is None:
                self.env_list[0].reset(backup=True)
                for idx in range(1, self.num_envs):
                    self.env_list[idx].set_state(self.env_list[0].state_copy)
            else:
                for idx in range(self.num_envs):
                    self.env_list[idx].set_state(dict_input)

            reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
            rate_sum = 0
            step_num = 80

            EPSILON = min(1 - 0.997 * math.exp((-0.0004)*episode), 0.8)                   # 探索率指数衰减

            if episode % 100 == 0:
                logging.info(f"Current episode: {episode}. EPSILON: {EPSILON}")

            for _ in range(step_num):                                                     # 开始一个episode (每一个循环代表一步)
                interact_idx = np.random.choice(self.num_envs, p=self.interact_distrib)
                
                state = self.env_list[interact_idx].get_state()
                
                action_code = self.model.choose_action(state, EPSILON)                                        # 输入该步对应的状态s，选择动作
                action = self.action_space.to_action_idx_list(action_code)
                next_state, reward, _ = self.env_list[interact_idx].step(action)

                self.model.buffers[interact_idx].store_transition(state, action, reward, next_state, False)                 # 存储样本
                reward_sum += reward
                rate = np.average(self.env_list[interact_idx].get_real_rate())
                rate_sum += rate
                state = next_state
                
                sample_idx = np.random.choice(self.num_envs, p=self.sample_distrib)
                if len(self.model.buffers[sample_idx]) == MEMORY_CAPACITY:
                    self.model.learn(sample_idx)

            reward_list.append(reward_sum / step_num)
            rate_list.append(rate_sum / step_num)
            
            if episode % 100 == 0:
                logging.info(f"Current average reward: {'%.3f' % (reward_sum / step_num)}")
                logging.info(f"Current transmission rate: {'%.4f' % (rate_sum / step_num)}")
                
        reward_address = f"./train_result/reward_list__{time.strftime('%Y-%m-%d_%H-%M-%S')}.npy"
        np.save(reward_address, reward_list)
        
        rate_address = f"./train_result/rate_list__{time.strftime('%Y-%m-%d_%H-%M-%S')}.npy"
        np.save(rate_address, rate_list)

        model_address = f"./train_result/{self.algorithm}_model_parameters__{time.strftime('%Y-%m-%d_%H-%M-%S')}.npy"
        torch.save(self.model.eval_net.state_dict(), model_address)


if __name__ == '__main__':
    agent = UAVAgent(
        env_list = [
            UAVEnvironment(sim=False)
        ]
    )

    user_loc = np.load('./user_loc.npy', allow_pickle=True)
    agent.train()
