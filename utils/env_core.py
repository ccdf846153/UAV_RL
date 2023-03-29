from utils.space import ActionSpace, StateSpace
from abc import abstractmethod
from typing import Optional


class Env(object):

    state_space: Optional[StateSpace] = None
    action_space: Optional[ActionSpace] = None

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    # not ready for visualization
    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError


