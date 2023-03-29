import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from typing import List

from modules.Net import Net
from utils.space import StateSpace, ActionSpace
from utils.buffer import ReplayBuffer


# 强化学习抽象方法类
class ReinforcementLearning(object):

    step_counter: int = 0

    @abstractmethod
    def choose_action(self):
        raise NotImplementedError
    
    @abstractmethod
    def learn(self):
        raise NotImplementedError


# DQN
class DQN(ReinforcementLearning):
    def __init__(
        self, 
        state_space: StateSpace = None,
        action_space: ActionSpace = None, 
        eval_net: nn.Module = None,
        target_net: nn.Module = None,
        gamma: float = 0.9,
        loss_function: nn.Module = None,
        optim_info: dict = {
            'net': None,
            'type': None,
            'lr': 0.001
        },
        buffer_info: dict = {
            'type': None,
            'capacity': 2000,
            'num': 4
        },
        batch_size: int = 32,
        target_replace_step: int = 100
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.eval_net = eval_net
        self.target_net = target_net
        self.loss_function = loss_function
        self.buffers: List[ReplayBuffer] = [
            eval(buffer_info['type'])(
                state_space = self.state_space,
                action_space = self.action_space,
                capacity = buffer_info['capacity']
            ) for _ in range(buffer_info['num'])
        ]
        self.num_buffers = buffer_info['num']
        self.step_counter = 0                                   # 学习步数
        self.optimizer: torch.optim.Optimizer = eval(f"torch.optim.{optim_info['type']}")(
            eval(f"self.{optim_info['net']}.parameters()"), 
            lr=optim_info['lr']
        )

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_replace_step = target_replace_step
    
    # 接收状态 state 和试探参数 EPSILON, 输出动作编码 (xxxxxx)_n, n 为单个对象动作空间大小, 位数为对象数量
    def choose_action(self, state, EPSILON) -> int:
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        # print(state.shape)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(state)
            action_code = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action_code = np.random.choice(len(self.action_space))
        return action_code

    # 学习函数
    # 整体流程包括：
    # 1. 每轮学习开始，从经验回放寄存器中提取待学习数据
    # 2. 动作评估网络和目标网络进行更新，步骤为：
    #     (1) 动作评估网络接收采样数据的所有开始状态，通过网络输出动作的 Q 值
    #     (2) 动作评估网络对采样动作进行 Q 值的提取与聚合
    #     (3) 动作目标网络接收采样数据的所有下一刻状态，并推测出这些状态对应所有动作的 Q 值
    #     (4) 动作目标网络得到采样一步后再次评估得到的 Q 值
    #     (5) 将原始输出的 Q 值与采样一步后合成的 Q 值作比较，并使用损失函数计算两个 Q 值之间的不一致程度
    #     (6) 将损失进行反向传播，并更新动作评估网络中的所有参数
    def learn(self, buffer_idx):
        if self.step_counter % self.target_replace_step == 0:                  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        self.step_counter += 1                                            # 学习步数自加1

        states, actions, rewards, next_states, _ = self.buffers[buffer_idx].sample(self.batch_size)

        q_eval = torch.gather(self.eval_net(states), dim=1, index=actions.view(len(actions), 1))
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(next_states).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = (rewards + self.gamma * torch.max(q_next, 1)[0]).view(self.batch_size, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_function(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数