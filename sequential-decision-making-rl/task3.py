import random
import numpy as np
from enum import Enum, auto
from typing import *
from gridworld import *
import time

class QLearningAgent:
    def __init__(self, gridworld: Gridworld, alpha: float = 0.1, epsilon: float = 0.1, 
                 gamma: float = 0.9, episodes: int = 20000):
        self.gridworld = gridworld
        self.alpha = alpha  
        self.epsilon = epsilon  
        self.gamma = gamma 
        self.episodes = episodes
        self.runtime = 0.0
        self.conv_episode = episodes
        
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
            best_actions = [a for a, v in self.q_values[state].items() if abs(v - max_val) < 1e-9]
            
            best_actions.sort(key=lambda x: str(x))
            return best_actions[0]

    def train(self):
        start_time = time.time()
        prev_policy = {} 
        stable_count = 0
        
        for ep in range(1, self.episodes + 1):
            state = (0, 0) 
            while True:
                actions = self.gridworld.get_actions(state)
                if not actions: break
                
                action = self.choose_action(state)
                transitions = self.gridworld.get_transitions(state, action)
                next_state = random.choices(list(transitions.keys()), weights=list(transitions.values()), k=1)[0]
                reward = self.gridworld.get_reward(state, action, next_state) 
                
                old_q = self.q_values[state][action]
                next_max = self.get_max_q(next_state)
                self.q_values[state][action] = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
                state = next_state

            # Get current greedy policy for all states
            current_policy = {s: self.getPolicy(s) for s in self.gridworld.states if self.gridworld.get_actions(s)}
            
            if current_policy == prev_policy:
                stable_count += 1
            else:
                stable_count = 0
                
            prev_policy = current_policy

            # If policy is stable for 300 episodes 
            if stable_count >= 300 and self.conv_episode == self.episodes:
                self.conv_episode = ep - 300

        self.runtime = time.time() - start_time

    def getPolicy(self, state: Gridworld.State):
        actions = self.gridworld.get_actions(state)
        if not actions: return None
        max_val = self.get_max_q(state)
        best_actions = [a for a, v in self.q_values[state].items() if abs(v - max_val) < 1e-9]
        
        best_actions.sort(key=lambda x: str(x)) 
        return best_actions[0]

    def getValue(self, state: Gridworld.State) -> float:
        actions = self.gridworld.get_actions(state)
        if not actions:
            # If terminal, get the actual reward value from the grid
            val = self.gridworld._grid_value(state)
            return float(val) if isinstance(val, (int, float)) else 0.0
        return self.get_max_q(state)

# --- Output Formatting ---
def print_grid_with_axes(agent, grid, label, mode="value"):
    n, m = len(grid), len(grid[0])
    symbols = {
        Gridworld.Action.Up: "↑",
        Gridworld.Action.Down: "↓",
        Gridworld.Action.Left: "←",
        Gridworld.Action.Right: "→",
        None: "X",
    }
    
    title = "Value Function" if mode == "value" else "Policy"
    print(f"\n{'='*65}")
    print(f" {title} — {label}")
    print(f"{'='*65}")
    
    header_offset = "        " 
    header = header_offset + "".join([f"y={i:<6}" for i in range(m)])
    print(header)
    print("   " + "-" * (len(header) - 3))

    for row_idx in range(n):
        x = n - 1 - row_idx
        row_str = f"x={x:<2} |   "
        
        for y in range(m):
            cell = grid[row_idx][y]
            if cell == '#':
                display = "XXXXXX  " if mode == "value" else "#       "
            elif isinstance(cell, float):
                display = " GOAL   " if mode == "value" else "G     "
            else:
                if mode == "value":
                    display = f"{agent.getValue((x, y)):+6.2f}  "
                else:
                    display = f"{symbols[a]:<8}" if (a := agent.getPolicy((x, y))) else "X       "
            
            row_str += display
        print(row_str)

def main():

    grid = (
        (' ', ' ', ' ', ' ', 10.), 
        (' ', ' ', ' ', ' ', ' '),  
        (' ', '#', ' ', '#', ' '),  
        (' ', ' ', ' ', ' ', ' '),  
        ('S', ' ', ' ', ' ', ' '),  
    )
    game = Gridworld(noise=0.2, living_reward=-1.0, grid=grid)

    ql = QLearningAgent(game, alpha=0.1, epsilon=0.1, gamma=0.9, episodes=20000)
    print("Training Q-Learning Agent...")
    ql.train()
    
    print_grid_with_axes(ql, grid, "Q-Learning", mode="value")
    print_grid_with_axes(ql, grid, "Q-Learning", mode="policy")

    print(f"\n{'='*70}")
    print(f"{'Algorithm':<20} | {'Total Episodes':<15} | {'Conv. Episode':<15} | {'Time (s)'}")
    print("-" * 70)
    print(f"{'Q-Learning':<20} | {ql.episodes:<15} | {ql.conv_episode:<15} | {ql.runtime:.4f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()