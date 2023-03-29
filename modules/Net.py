import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional


# actor-critic
class Actor(nn.Module):
    def __init__(self, optim_action, dim_obs_space, dim_goal, dim_action_space):
        super(Actor, self).__init__()
        self.optim_action = optim_action
        
        self.fc1 = nn.Linear(dim_obs_space + dim_goal, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.next_action = nn.Linear(64, dim_action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.optim_action * torch.sigmoid(self.next_action(x))

        return actions
    
class Critic(nn.Module):
    def __init__(self, optim_action, dim_obs_space, dim_goal, dim_action_space):
        super(Critic, self).__init__()
        self.optim_action = optim_action
        self.fc1 = nn.Linear(dim_obs_space + dim_goal + dim_action_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_eval_out = nn.Linear(64, 1)
    
    def forward(self, x, actions):
        x = torch.cat([x, actions / self.optim_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_eval = self.action_eval_out(x)

        return action_eval
    
# 网络结构定义
class Net(nn.Module):
    def __init__(self, num_input, num_output):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(num_input, 64)                                      
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_output)                                     

    def forward(self, x):                                                # 定义forward函数 (x为状态)
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值