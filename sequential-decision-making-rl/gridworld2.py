from enum import Enum, auto
from typing import *

class Gridworld:
    State = Tuple[int, int]

    class Action(Enum):
        Up = auto()
        Down = auto()
        Left = auto()
        Right = auto()

    def __init__(self, noise: float, living_reward: float, grid: Tuple[Tuple[Any, ...], ...]):
        self.__noise = noise
        self.__living_reward = living_reward
        self.__n = len(grid)
        self.__m = len(grid[0])
        self.__grid = grid
        self.__states = {(x, y) for x in range(self.__n) for y in range(self.__m) if
                         grid[self.__n - 1 -x][y] in (' ', 'S')}

    @property
    def states(self) -> Set[State]:
        return self.__states

    def get_actions(self, state: State) -> Set[Action]:
        x, y = state
        if x < 0 or x >= self.__n or y < 0 or y >= self.__m:
            raise ValueError('Not a valid state')
        if isinstance(self._grid_value(state), float):  # Return no actions if terminal state
            return set()
        return {*Gridworld.Action}
    
    def _grid_value(self, state: State):
        x, y = state
        return self.__grid[self.__n - 1 - x][y]

    def _do_action(self, state: State, action: Action) -> State: # Returns the new state
        x, y = state

        if action == Gridworld.Action.Up:
            target_x, target_y = x + 1, y
        elif action == Gridworld.Action.Down:
            target_x, target_y = x - 1, y
        elif action == Gridworld.Action.Left:
            target_x, target_y = x, y - 1
        else:
            target_x, target_y = x, y + 1

        if target_x < 0 or target_x >= self.__n or target_y < 0 or target_y >= self.__m or \
                self.__grid[self.__n - 1 - target_x][target_y] == '#':
            return state
        return target_x, target_y

    def get_transitions(self, current_state: State, action: Action) -> Dict[State, float]: # Returns a dictionary of "next_state": probability
        if action not in self.get_actions(current_state):
            raise ValueError('not a valid action')

        if self.__noise <= 0.:
            return {self._do_action(current_state, action): 1.}

        remaining = self.__noise / 2.
        if action in (Gridworld.Action.Up, Gridworld.Action.Down):
            outcomes = (
                (self._do_action(current_state, action), 1 - self.__noise),
                (self._do_action(current_state, Gridworld.Action.Left), remaining),
                (self._do_action(current_state, Gridworld.Action.Right), remaining)
            )
        else:
            outcomes = (
                (self._do_action(current_state, action), 1 - self.__noise),
                (self._do_action(current_state, Gridworld.Action.Up), remaining),
                (self._do_action(current_state, Gridworld.Action.Down), remaining)
            )

        transitions = {}
        for outcome, val in outcomes:
            transitions[outcome] = transitions.get(outcome, 0.) + val
        return transitions

    def get_reward(self, current_state: State, action: Action, next_state: State) -> float:
        if next_state not in self.get_transitions(current_state, action):
            raise ValueError('next state is not reachable from current state')
        grid_value = self._grid_value(next_state)
        return grid_value if isinstance(grid_value, float) else self.__living_reward