import random
import numpy as np
from enum import Enum, auto
from typing import *
from gridworld import *


# --- Monte Carlo Agent Implementation ---

class MonteCarloAgent:
    def __init__(self, gridworld: 'Gridworld', epsilon: float = 0.2, 
                 gamma: float = 0.9, episodes: int = 20000):
        self.gridworld = gridworld
        self.epsilon = epsilon  
        self.gamma = gamma 
        self.episodes = episodes
        
        # Q-table: stores expected returns for each action in each state
        self.q_values = {}
        # Track returns for averaging
        self.returns_sum = {}
        self.returns_count = {}
        
        for state in self.gridworld.states:
            actions = self.gridworld.get_actions(state)
            self.q_values[state] = {a: 0.0 for a in actions}
            self.returns_sum[state] = {a: 0.0 for a in actions}
            self.returns_count[state] = {a: 0.0 for a in actions}

    def get_max_q(self, state: 'Gridworld.State') -> float:
        actions = self.gridworld.get_actions(state)
        if not actions:
            return 0.0
        return max(self.q_values[state].values())

    def choose_action(self, state: 'Gridworld.State') -> 'Gridworld.Action':
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
            # 1. Generate an episode
            episode = []
            state = start_state
            
            # Step limit to prevent infinite loops during early training
            for _step in range(100):
                actions = self.gridworld.get_actions(state)
                if not actions: 
                    break
                
                action = self.choose_action(state)
                
                # Sample environment
                transitions = self.gridworld.get_transitions(state, action)
                next_state = random.choices(list(transitions.keys()), 
                                            weights=list(transitions.values()), k=1)[0]
                
                reward = self.gridworld.get_reward(state, action, next_state) 
                episode.append((state, action, reward))
                state = next_state

            # 2. Process episode (First-Visit Monte Carlo)
            g = 0 # Return
            visited_state_actions = set()
            
            # Walk backwards through the episode
            for i in range(len(episode) - 1, -1, -1):
                s, a, r = episode[i]
                g = r + self.gamma * g
                
                if (s, a) not in visited_state_actions:
                    visited_state_actions.add((s, a))
                    self.returns_sum[s][a] += g
                    self.returns_count[s][a] += 1
                    # Average the returns to update Q-value
                    self.q_values[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]

    def getPolicy(self, state: 'Gridworld.State'):
        actions = self.gridworld.get_actions(state)
        if not actions: return None
        max_val = self.get_max_q(state)
        best_actions = [a for a, v in self.q_values[state].items() if v == max_val]
        return random.choice(best_actions)

    def getValue(self, state: 'Gridworld.State') -> float:
        actions = self.gridworld.get_actions(state)
        if not actions:
            val = self.gridworld._grid_value(state)
            return float(val) if isinstance(val, (int, float)) else 0.0
        return self.get_max_q(state)

# --- Output Formatting ---

def print_value_function(agent: MonteCarloAgent, grid: tuple, label: str):
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
                row += "   ##### "
            elif isinstance(cell, float):
                row += f"   GOAL  "
            else:
                row += f" {agent.getValue((x, y)):+7.2f} "
        print(row)

def print_policy(agent: MonteCarloAgent, grid: tuple, label: str):
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

# --- Main Logic ---

def main():

    grid = (
        (' ', ' ', ' ', ' ', 10.), 
        (' ', ' ', ' ', ' ', ' '),  
        (' ', '#', ' ', '#', ' '),  
        (' ', ' ', ' ', ' ', ' '),  
        ('S', ' ', ' ', ' ', ' '),  
    )
    game = Gridworld(noise=0.2, living_reward=-1.0, grid=grid)

    # Initialize and Train Monte Carlo Agent
    mc = MonteCarloAgent(game, epsilon=0.2, gamma=0.9, episodes=50000)
    print("Training Monte Carlo Agent...")
    mc.train()
    
    print_value_function(mc, grid, "Monte Carlo")
    print_policy(mc, grid, "Monte Carlo")

if __name__ == '__main__':
    main()