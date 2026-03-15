import random
import numpy as np
from enum import Enum, auto
from typing import *
from gridworld2 import *

class QLearningAgent:
    def __init__(self, gridworld: Gridworld, alpha: float = 0.1, epsilon: float = 0.1, 
                 gamma: float = 0.9, episodes: int = 20000):
        self.gridworld = gridworld
        self.alpha = alpha  
        self.epsilon = epsilon  
        self.gamma = gamma 
        self.episodes = episodes
        
        # Initialize Q-table using the states set provided by Gridworld
        self.q_values = {}
        for state in self.gridworld.states:
            actions = self.gridworld.get_actions(state)
            self.q_values[state] = {a: 0.0 for a in actions}

    def get_max_q(self, state: Gridworld.State) -> float:
        actions = self.gridworld.get_actions(state)
        if not actions:
            return 0.0
        return max(self.q_values[state].values())

    def choose_action(self, state: Gridworld.State) -> Gridworld.Action:
        actions = list(self.gridworld.get_actions(state))
        if not actions:
            return None
        
        # epsilon-greedy strategy 
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            max_val = self.get_max_q(state)
            best_actions = [a for a, v in self.q_values[state].items() if v == max_val]
            return random.choice(best_actions)

    def train(self):
        start_state = (0, 0) 
        
        for _ in range(self.episodes):
            state = start_state
            while True:
                actions = self.gridworld.get_actions(state)
                if not actions: # Terminal state reached
                    break
                
                action = self.choose_action(state)
                
                # Sample next state based on environment transitions
                transitions = self.gridworld.get_transitions(state, action)
                next_states = list(transitions.keys())
                probs = list(transitions.values())
                next_state = random.choices(next_states, weights=probs, k=1)[0]
                
                reward = self.gridworld.get_reward(state, action, next_state) 
                
                # Q-Learning update rule
                old_q = self.q_values[state][action]
                next_max = self.get_max_q(next_state)
                self.q_values[state][action] = old_q + self.alpha * (
                    reward + self.gamma * next_max - old_q
                )
                state = next_state

    def getPolicy(self, state: Gridworld.State):
        actions = self.gridworld.get_actions(state)
        if not actions: return None
        max_val = self.get_max_q(state)
        best_actions = [a for a, v in self.q_values[state].items() if v == max_val]
        return random.choice(best_actions)

    def getValue(self, state: Gridworld.State) -> float:
        actions = self.gridworld.get_actions(state)
        if not actions:
            # If terminal, get the actual reward value from the grid
            val = self.gridworld._grid_value(state)
            return float(val) if isinstance(val, (int, float)) else 0.0
        return self.get_max_q(state)

# --- Output Formatting ---
def print_value_function(agent: QLearningAgent, grid: tuple, label: str):
    n, m = len(grid), len(grid[0])
    print(f"\n{'='*55}")
    print(f"Value Function — {label}")
    print(f"{'='*55}")
    for r in range(n):
        row = ""
        for c in range(m):
            x, y = n - 1 - r, c
            cell = grid[r][c]
            if cell == '#':
                row += "  #####  "
            elif isinstance(cell, float):
                row += f"   GOAL "
            else:
                row += f" {agent.getValue((x, y)):+7.2f} "
        print(row)

def print_policy(agent: QLearningAgent, grid: tuple, label: str):
    symbols = {
        Gridworld.Action.Up: "↑", Gridworld.Action.Down: "↓",
        Gridworld.Action.Left: "←", Gridworld.Action.Right: "→",
        None: "X",
    }
    n, m = len(grid), len(grid[0])
    print(f"\n{'='*55}")
    print(f"Policy — {label}")
    print(f"{'='*55}")
    for r in range(n):
        row = ""
        for c in range(m):
            x, y = n - 1 - r, c
            cell = grid[r][c]
            if cell == '#':
                row += "  # "
            elif isinstance(cell, float):
                row += "  G "
            else:
                a = agent.getPolicy((x, y))
                row += f"  {symbols[a]} "
        print(row)

def main():
    RANDOM_SEED = 42 
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    grid = (
        (' ', ' ', ' ', ' ', 10.), 
        (' ', ' ', ' ', ' ', ' '),  
        (' ', '#', ' ', '#', ' '),  
        (' ', ' ', ' ', ' ', ' '),  
        ('S', ' ', ' ', ' ', ' '),  
    )
    game = Gridworld(noise=0.2, living_reward=-1.0, grid=grid)

    ql = QLearningAgent(game, alpha=0.1, epsilon=0.1, gamma=0.9, episodes=20000)
    ql.train()
    
    print_value_function(ql, grid, "Q-Learning")
    print_policy(ql, grid, "Q-Learning")

if __name__ == '__main__':
    main()