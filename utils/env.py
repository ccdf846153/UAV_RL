import random
import numpy as np
import matplotlib.pyplot as plt

from utils.env_core import Env
from utils.env_args import *
from utils.space import StateSpace, ActionSpace


class UAVEnvironment(Env):
    def __init__(
            self, 
            mode = 'discrete', 
            num_users = n_user, 
            num_uavs = n_uav, 
            drone_radius = 30, 
            map_size = (100, 100), 
            height = 5,
            speed = 4,
            unit_len = 1,
            sim=True
    ):
        # 环境参数
        self.action_space = ActionSpace(
            num_objects=num_uavs,
            num_choices=None,
            action_list=ACTION_DICT['discrete']['4-directions'],
            mode=mode
        )
        self.state_space = StateSpace(
            num_objects=num_uavs,
            num_states=None,
            state_list=STATE_DICT(map_size)['continuous'],
            mode='continuous'
        )
        self.num_users = num_users
        self.num_uavs = num_uavs
        self.drone_radius = drone_radius            # 无人机的覆盖半径（单位：格）
        self.map_size = map_size                    # 环境在 2D 上的网格划分
        self.height = height                        # 无人机的飞行所在高度（单位：格）
        self.speed = speed
        self.unit_len = unit_len                    # 无人机环境单元格在真实环境中的长度（单位：米）
        
        # 数据链路参数
        self.A = 1e-5                       # A = g_0 * (L_0)^theta 衰落常量（单位：m^2）
        self.theta = 2.8                    # 无线信道衰落参数
        self.p = 100                        # 发射功率（单位：mW）
        self.total_band = 10e7              # 无人机总传输带宽
        self.band = self.total_band / num_users
        self.noise = pow(10,-174/10) * self.band            # 信道噪声功率

        # 状态变量及环境动态属性
        self.state_space.add_property('user_loc_list', np.zeros((num_users, 2)))
        self.state_space.add_property('uav_loc_list', np.zeros((num_uavs, 2)))
        self.state_space.add_property('uav_user_distances', np.zeros((num_uavs, num_users)))
        self.state_space.add_property('cover_list', np.zeros(num_users))
        self.size = {
            'user_loc_list': (num_users, 2),
            'uav_loc_list': (num_uavs, 2),
            'uav_user_distances': (num_uavs, num_users),
            'cover_list': num_users
        }

        self.state_copy = {}
        self.copy_state()
        self.sim = sim
        self.dt_noise = DT_NOISE


    @property
    def num_actions(self):
        return len(self.action_space)
    
    @property
    def num_states(self):
        return self.state_space.num_objects * self.state_space.num_state_types

    # 重置环境状态
    def reset(self, random=True, backup=False):
        if random:
            self.state_space.uav_loc_list = np.array(
                [np.random.uniform(low=0, high=self.map_size[0], size=self.num_uavs),
                np.random.uniform(low=0, high=self.map_size[1], size=self.num_uavs)]
            ).transpose()
            self.state_space.user_loc_list = np.array(
                [np.random.uniform(low=0, high=self.map_size[0], size=self.num_users),
                np.random.uniform(low=0, high=self.map_size[1], size=self.num_users)]
            ).transpose()
            self.state_space.cover_list = np.zeros(self.size['cover_list'])
            self._update_distances()
            if backup:
                self.copy_state()
        else:
            for property in self.state_copy:
                setattr(self.state_space, property, self.state_copy[property])
    
    def copy_state(self):
        self.state_copy['user_loc_list'] = self.state_space.user_loc_list.copy()
        self.state_copy['uav_loc_list'] = self.state_space.uav_loc_list.copy()
        self.state_copy['uav_user_distances'] = self.state_space.uav_user_distances.copy()
        self.state_copy['cover_list'] = self.state_space.cover_list.copy()

    def set_state(self, state_dict):
        self.state_copy['user_loc_list'] = state_dict['user_loc_list'].copy()
        self.state_copy['uav_loc_list'] = state_dict['uav_loc_list'].copy()
        self.state_copy['uav_user_distances'] = state_dict['uav_user_distances'].copy()
        self.state_copy['cover_list'] = state_dict['cover_list'].copy()
        
    # 更新距离列表
    def _update_distances(self):
        for drone_idx, drone_loc in enumerate(self.state_space.uav_loc_list):
            for user_idx, user_loc in enumerate(self.state_space.user_loc_list):
                self.state_space.uav_user_distances[drone_idx][user_idx] = pow(
                        pow(self.height * self.unit_len, 2) + 
                        pow((drone_loc[0] - user_loc[0]) * self.unit_len, 2) + 
                        pow((drone_loc[1] - user_loc[1]) * self.unit_len, 2), 
                    0.5)
    
    # 更新数据速率
    def _update_transmission_rate(self, sim=None):
        # snr = (self.A * self.p) / (pow(self.state_space.uav_user_distances, self.theta) * self.noise)
        # channel_capacities = self.band * np.log2(1 + snr)
        # trans_rates = np.zeros(self.num_users)
        # for user_idx in range(self.num_users):
        #     trans_rates[user_idx] = np.max(channel_capacities[:, user_idx])
        # return trans_rates / 5e6
        snr = (self.A * self.p) / (pow(self.state_space.uav_user_distances, self.theta) * self.noise)
        channel_capacities = self.band * np.log2(1 + snr)
        trans_rates = np.zeros(self.num_users)
        if sim is None and self.sim:
            noise_power = self.dt_noise * channel_capacities
            channel_capacities += np.random.normal(loc=0,scale=noise_power,size=(self.num_uavs, self.num_users))
        for user_idx in range(self.num_users):
            trans_rates[user_idx] = np.max(channel_capacities[:, user_idx])
        return trans_rates / 5e6
    
    def get_real_rate(self):
        return self._update_transmission_rate(sim=False)
    
    # def get_sim_rate(self):
    #     snr = (self.A * self.p) / (pow(self.state_space.uav_user_distances, self.theta) * self.noise)
    #     channel_capacities = self.band * np.log2(1 + snr)
    #     trans_rates = np.zeros(self.num_users)
    #     noise_power = self.dt_noise * channel_capacities
    #     trans_rates += np.array([[random.gauss(0, sigma=noise_power) for _ in self.num_users] for _ in self.num_uavs])
    #     for user_idx in range(self.num_users):
    #         trans_rates[user_idx] = np.max(channel_capacities[:, user_idx])
    #     return trans_rates / 5e6

    # 奖励生成函数
    def reward(self):
        self._update_distances()
        trans_rates = self._update_transmission_rate()
        return np.average(trans_rates)

    # 返回当前状态
    def get_state(self):
        return self.state_space.uav_loc_list.flatten()
    
    # 执行动作并返回奖励和下一状态
    def step(self, action_idx_list):
        # 执行无人机运动动作
        actions = self.action_space.to_action_list(action_idx_list)
        high = [self.map_size[0] - self.speed // 2, self.map_size[1] - self.speed // 2]
        low = [self.speed // 2, self.speed // 2]

        for idx in range(self.num_uavs):
            self.state_space.uav_loc_list[idx][0] += actions[idx][0] * self.speed
            self.state_space.uav_loc_list[idx][0] += actions[idx][1] * self.speed
            if self.state_space.uav_loc_list[idx][0] > high[0]:
                self.state_space.uav_loc_list[idx][0] = high[0]
            if self.state_space.uav_loc_list[idx][0] < low[0]:
                self.state_space.uav_loc_list[idx][0] = low[0]
            if self.state_space.uav_loc_list[idx][1] > high[1]:
                self.state_space.uav_loc_list[idx][1] = high[1]
            if self.state_space.uav_loc_list[idx][1] < low[1]:
                self.state_space.uav_loc_list[idx][1] = low[1]

        rewards = self.reward()
        
    #     # 计算覆盖率和奖励
    #     # reward = 0
    #     # for i in range(self.num_users):
    #     #     for j in range(self.num_drones):
    #     #         dist = np.linalg.norm(self.drones[j] - self.users[i])
    #     #         if dist < self.drone_radius:
    #     #             self.coverage[i] = 1
    #     #             reward += 1 - dist / self.drone_radius
        
    #     # 判断是否所有用户都得到了服务
    #     # done = (np.sum(self.coverage) == self.num_users)
        
        return self.get_state(), rewards, False
    
    def test_reset(self):
        # print(self.drones)
        # print(self.users)
        plt.scatter(self.state_space.uav_loc_list[:,0], self.state_space.uav_loc_list[:,1])
        plt.scatter(self.state_space.user_loc_list[:,0], self.state_space.user_loc_list[:,1])
        plt.xlim(-5, 105)
        plt.ylim(-5, 105)
        plt.show()




if __name__ == '__main__':
    env = UAVEnvironment()
    # env.reset()
    
    # import inspect
    # print(env.state_space.__dict__.keys())
    # env.test_reset()
    