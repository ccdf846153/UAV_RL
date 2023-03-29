import numpy as np
from typing import List, Dict
from abc import abstractmethod


class Space(object):
    @abstractmethod
    def sample(self):
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    def _continuous_interval_sample(self, left=0., right=1., include_left=True, include_right=False):
        if right < left:
            raise ValueError("Not a valid interval. 'right' greater than/equal 'left' expected.")
        if right == left:
            return left
        interval = right - left
        if include_left == include_right:
            temp_random = [np.random.sample(), np.random.sample()]
            if temp_random[0] == temp_random[1] and temp_random[0] == 0:
                return np.random.randint(0, 2)[0] * interval + left
            else:
                return (temp_random[0] + temp_random[1]) * interval / 2 + left
        else:
            if include_left:
                return np.random.sample() * interval + left
            else:
                return right - np.random.sample() * interval


class ActionSpace(Space):
    def __init__(self, num_objects, num_choices=None, action_list=None, action_type='default', mode='discrete'):
        self.type = mode
        self.num_objects = num_objects
        if mode == 'discrete':
            if action_list is None:
                try:
                    assert num_choices is not None
                    self.num_choices = num_choices
                    if action_type == 'default':
                        self.action_list = list(range(num_choices))
                    else:
                        self.action_list = [-1] * num_choices
                except AssertionError:
                    raise ValueError("'num_choices' and 'action_list' cannot both be NoneType value.")
            else:
                self.action_list = action_list
                self.num_choices = len(action_list)
        elif mode == 'continuous':
            try:
                assert action_list is not None and isinstance(action_list, dict)
                for property in action_list:
                    try:
                        assert isinstance(action_list[property], list)
                    except AssertionError:
                        raise ValueError(f"'action_list['{property}']' not a valid value.")
            except AssertionError:
                raise TypeError("'action_list' cannot be NoneType value.")
            self.action_list = [[value_range[idx][0:2] for idx in range(len(value_range))][0] for value_range in action_list.values()]
            self.num_action_types = len(self.action_list)
            self.bound_include = [[value_range[idx][2] for idx in range(len(value_range))][0] for value_range in action_list.values()]
        elif mode == 'mix':
            self.action_list = {}
            if action_list is None or not isinstance(action_list, dict):
                raise TypeError("'action_list' should be DictType value.")
            if 'discrete' not in action_list.keys() and 'continuous' not in action_list.keys():
                raise ValueError("'action_list' not a valid value. Expect keys 'discrete' and 'continuous'.")
            self.action_list['discrete'] = action_list['discrete']
            self.num_choices = len(self.action_list['discrete'])
            continuous_action_dict = action_list['continuous']
            try:
                assert continuous_action_dict is not None and isinstance(continuous_action_dict, dict)
                for property in continuous_action_dict:
                    try:
                        assert isinstance(continuous_action_dict[property], list)
                    except AssertionError:
                        raise ValueError(f"'action_list['continuous']['{property}']' not a valid value.")
            except AssertionError:
                raise TypeError("'action_list['continuous']' cannot be NoneType value. Maybe 'discrete' mode?")
            self.action_list['continuous'] = [[value_range[idx][0:2] for idx in range(len(value_range))][0] for value_range in continuous_action_dict.values()]
            self.num_action_types = len(continuous_action_dict)
            self.bound_include = [[value_range[idx][2] for idx in range(len(value_range))][0] for value_range in continuous_action_dict.values()]
        else:
            raise ValueError("'mode' not a valid input.")
    
    def __len__(self) -> int:
        if self.type == 'discrete':
            return self.num_choices ** self.num_objects
        else:
            return -1
    
    def __discrete_sample(self, p=None) -> List[int]:
        distrib = [1 / self.num_choices] * self.num_choices if p is None else p
        return np.random.choice(self.num_choices, self.num_objects, p=distrib).tolist()
    
    def __continuous_sample(self) -> List[float]:
        if self.type == 'continuous':
            return [self._continuous_interval_sample(*self.action_list[idx], *self.bound_include[idx])\
                    for idx in range(self.num_action_types)]
        elif self.type == 'mix':
            return [self._continuous_interval_sample(*self.action_list['continuous'][idx], *self.bound_include[idx])\
                    for idx in range(self.num_action_types)]
    
    def sample(self, p=None, action_idx=True):
        if self.type == 'discrete':
            if action_idx:
                return self.__discrete_sample(p)
            else:
                return self.to_action_list(self.__discrete_sample(p))
        elif self.type == 'continuous':
            return self.__continuous_sample()
        else:
            if action_idx:
                return [self.__discrete_sample(p), self.__continuous_sample()]
            else:
                return [self.to_action_list(self.__discrete_sample(p)), self.__continuous_sample()]

    # 接受输入：[action_idx1 action_idx2 ... action_idxn], 长度为 num_objects 的列表，各元素取值为 [0, num_choices)
    def to_action_list(self, action_idx_list: List[int]) -> List:
        try:
            assert len(action_idx_list) == self.num_objects
            if self.type == 'discrete':
                return [self.action_list[idx] for idx in action_idx_list]
            elif self.type == 'mix':
                return [self.action_list['discrete'][idx] for idx in action_idx_list]
        except AssertionError:
            raise IndexError("'action_idx_list' not correct size.")
    
    def _int_to_str(self, num, base, num_bits):
        result = ''
        while num > 0:
            result = f'{num % base}' + result
            num //= base
        return result if len(result) == num_bits else '0'*(num_bits-len(result)) + result
    
    def to_action_idx_list(self, actions) -> List[int]:
        if self.type == 'continuous':
            raise NotImplementedError("not valid in 'continuous' action space.")
        if isinstance(actions, list):
            idx_list = []
            if self.type == 'discrete':
                for action in actions:
                    for idx in range(self.num_choices):
                        if self.action_list[idx] == action:
                            idx_list.append(idx)
                            break
                    if idx == idx_list:
                        idx_list.append(-1)
            elif self.type == 'mix':
                for action in actions:
                    for idx in range(self.num_choices):
                        if self.action_list['discrete'][idx] == action:
                            idx_list.append(idx)
                            break
                    if idx == idx_list:
                        idx_list.append(-1)
            return idx_list
        else:
            if self.type == 'discrete' or self.type == 'mix':
                return list(map(int, self._int_to_str(actions, self.num_choices, self.num_objects)))

    def to_code(self, actions: List[int]):
        code = 0
        for idx in range(self.num_objects):
            code += actions[idx] * (self.num_choices ** (self.num_objects - idx - 1))
        return code

    def to_string(self, actions):
        pass


class StateSpace(Space):
    def __init__(self, num_objects, num_states=None, state_list=None, state_type='default', mode='discrete'):
        self.type = mode
        self.num_objects = num_objects
        if mode == 'discrete':
            if state_list is None:
                try:
                    assert num_states is not None
                    self.num_states = num_states
                    if state_type == 'default':
                        self.state_list = list(range(num_states))
                    else:
                        self.state_list = [-1] * num_states
                except AssertionError:
                    raise ValueError("'num_states' and 'state_list' cannot both be NoneType value.")
            else:
                self.state_list = state_list
                self.num_states = len(state_list)
        elif mode == 'continuous':
            try:
                assert state_list is not None and isinstance(state_list, dict)
                for property in state_list:
                    try:
                        assert isinstance(state_list[property], list)
                    except AssertionError:
                        raise ValueError(f"'state_list['{property}']' not a valid value.")
            except AssertionError:
                raise TypeError("'state_list' cannot be NoneType value.")
            self.state_list = [[value_range[idx][0:2] for idx in range(len(value_range))][0] for value_range in state_list.values()]
            self.num_state_types = len(self.state_list)
            self.bound_include = [[value_range[idx][2] for idx in range(len(value_range))][0] for value_range in state_list.values()]
        elif mode == 'mix':
            self.state_list = {}
            if state_list is None or not isinstance(state_list, dict):
                raise TypeError("'state_list' should be DictType value.")
            if 'discrete' not in state_list.keys() and 'continuous' not in state_list.keys():
                raise ValueError("'state_list' not a valid value. Expect keys 'discrete' and 'continuous'.")
            self.state_list['discrete'] = state_list['discrete']
            self.num_states = len(self.state_list['discrete'])
            continuous_state_dict = state_list['continuous']
            try:
                assert continuous_state_dict is not None and isinstance(continuous_state_dict, dict)
                for property in continuous_state_dict:
                    try:
                        assert isinstance(continuous_state_dict[property], list)
                    except AssertionError:
                        raise ValueError(f"'state_list['continuous']['{property}']' not a valid value.")
            except AssertionError:
                raise TypeError("'state_list['continuous']' cannot be NoneType value. Maybe 'discrete' mode?")
            self.state_list['continuous'] = [[value_range[idx][0:2] for idx in range(len(value_range))][0] for value_range in continuous_state_dict.values()]
            self.num_state_types = len(continuous_state_dict)
            self.bound_include = [[value_range[idx][2] for idx in range(len(value_range))][0] for value_range in continuous_state_dict.values()]
        else:
            raise ValueError("'mode' not a valid input.")
    
    def __len__(self) -> int:
        if self.type == 'discrete':
            return self.num_states ** self.num_objects
        elif self.type == 'continuous':
            return self.num_state_types * self.num_objects
    
    def __discrete_sample(self, p=None) -> List[int]:
        distrib = [1 / self.num_states] * self.num_states if p is None else p
        return np.random.choice(self.num_states, self.num_objects, p=distrib).tolist()
    
    def __continuous_sample(self) -> List[float]:
        if self.type == 'continuous':
            return [self._continuous_interval_sample(*self.state_list[idx], *self.bound_include[idx])\
                    for idx in range(self.num_state_types)]
        elif self.type == 'mix':
            return [self._continuous_interval_sample(*self.state_list['continuous'][idx], *self.bound_include[idx])\
                    for idx in range(self.num_state_types)]
    
    def sample(self, p=None, state_idx=True):
        if self.type == 'discrete':
            if state_idx:
                return self.__discrete_sample(p)
            else:
                return self.to_state_list(self.__discrete_sample(p))
        elif self.type == 'continuous':
            return self.__continuous_sample()
        else:
            if state_idx:
                return [self.__discrete_sample(p), self.__continuous_sample()]
            else:
                return [self.to_state_list(self.__discrete_sample(p)), self.__continuous_sample()]

    # 接受输入：[state_idx1 state_idx2 ... state_idxn], 长度为 num_objects 的列表，各元素取值为 [0, num_states)
    def to_state_list(self, state_idx_list: List[int]) -> List:
        try:
            assert len(state_idx_list) == self.num_objects
            if self.type == 'discrete':
                return [self.state_list[idx] for idx in state_idx_list]
            elif self.type == 'mix':
                return [self.state_list['discrete'][idx] for idx in state_idx_list]
        except AssertionError:
            raise IndexError("'state_idx_list' not correct size.")
    
    def add_property(self, property_name=None, property_list=None):
        try:
            assert isinstance(property_name, str) and property_list is not None
            assert isinstance(property_list, list) or isinstance(property_list, np.ndarray)
        except AssertionError:
            raise TypeError("'property_name' should be str object and 'property_list' should be ListType or numpy.ndarray object.")
        if not hasattr(self, property_name):
            setattr(self, property_name, property_list)
    
    def get_state_idx(self, state):
        if self.type == 'continuous':
            raise NotImplementedError("not valid in 'continuous' mode state space.")
        for idx in range(self.num_states):
            if self.type == 'discrete' and state == self.state_list[idx]:
                return idx
            elif self.type == 'mix' and state == self.state_list['discrete'][idx]:
                return idx
        return -1
    
    def set_object_states(self, state_list):
        self.object_states = state_list

    def to_string(self, states):
        pass






# * 测试代码
# if __name__ == '__main__':
    discrete_action_space = ActionSpace(
        num_objects=4,
        num_choices=None,
        action_list=[(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
        mode='discrete'
    )

#     continuous_action_space = ActionSpace(
#         num_objects=4,
#         action_list= {
#             'angle': [[-180, 180, [True, False]]],
#             'distance': [[0, 3, [True, True]]]
#         },
#         mode='continuous'
#     )

#     mix_action_space = ActionSpace(
#         num_objects=4,
#         action_list= {
#             'discrete': [-1, -2, -3, -4, -5, -6],
#             'continuous': {
#                 'angle': [[-180, 180, [True, False]]],
#                 'distance': [[0, 3, [True, True]]]
#             }
#         },
#         mode='mix'
#     )

#     print(discrete_action_space.action_list, discrete_action_space.num_choices)
#     print(continuous_action_space.action_list, continuous_action_space.num_action_types, continuous_action_space.bound_include)
#     print(mix_action_space.action_list['discrete'], mix_action_space.action_list['continuous'])
#     print(mix_action_space.num_choices, mix_action_space.num_action_types, mix_action_space.bound_include)
#     print(discrete_action_space.sample(), continuous_action_space.sample(), mix_action_space.sample(), sep='\n')
#     print(discrete_action_space.sample(action_idx=False), mix_action_space.sample(action_idx=False), sep='\n')

    # print(len(discrete_action_space))
    # choice = np.random.choice(len(discrete_action_space))
    # print(choice, discrete_action_space.to_action_idx_list(choice))
    # choice_list = discrete_action_space.sample()
    # num = discrete_action_space.to_code(choice_list)
    # print(choice_list)
    # print(num)
    # print(ActionSpace._int_to_str(discrete_action_space, num, discrete_action_space.num_choices, discrete_action_space.num_objects))
    # action_list = discrete_action_space.to_action_list(choice_list)
    # print(choice_list, action_list, discrete_action_space.to_action_idx_list(action_list))
