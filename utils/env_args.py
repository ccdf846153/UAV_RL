n_uav = 4
n_user = 100
map_size = (100, 100)
# 超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.001                                       # 学习率
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
N_ACTIONS = 4**n_uav                
N_STATES = 2*n_uav
DT_NOISE = 0.3

ACTION_DICT = {
    'discrete': {
        '4-directions': [(1, 0), (-1, 0), (0, 1), (0, -1)],
    }
}

def STATE_DICT(map_size):
    return {
        'continuous': {
            'x': [[0, map_size[0], [True, True]]],
            'y': [[0, map_size[1], [True, True]]]
        }
    }