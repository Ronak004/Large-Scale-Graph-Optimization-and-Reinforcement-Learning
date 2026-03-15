import random
import numpy as np
from enum import Enum, auto
from typing import *
from gridworld import *
import time


# --- Monte Carlo Agent Implementation ---

class MonteCarloAgent:
    def __init__(self, gridworld: 'Gridworld', epsilon: float = 0.2, 
                 gamma: float = 0.9, episodes: int = 20000):
        self.gridworld = gridworld
        self.epsilon = epsilon  
        self.gamma = gamma 
        self.episodes = episodes
        self.runtime = 0.0
        self.conv_episode = episodes # Default to max if no convergence found
        
        self.q_values = {state: {a: 0.0 for a in gridworld.get_actions(state)} for state in gridworld.states}
        self.returns_sum = {state: {a: 0.0 for a in gridworld.get_actions(state)} for state in gridworld.states}
        self.returns_count = {state: {a: 0.0 for a in gridworld.get_actions(state)} for state in gridworld.states}

    def get_max_q(self, state: 'Gridworld.State') -> float:
        actions = self.gridworld.get_actions(state)
        if not actions: return 0.0
        return max(self.q_values[state].values())

    def choose_action(self, state: 'Gridworld.State') -> 'Gridworld.Action':
        actions = list(self.gridworld.get_actions(state))
        if not actions: return None
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            max_val = self.get_max_q(state)
            best_actions = [a for a, v in self.q_values[state].items() if abs(v - max_val) < 1e-9]
            best_actions.sort(key=lambda x: str(x))
            return best_actions[0]

    def train(self):
        start_time = time.time()
        start_state = (0, 0) 
        prev_policy = {} 
        stable_count = 0 
        
        for ep in range(1, self.episodes + 1):
            episode_history = []
            state = start_state
            
            # 1. Generate Episode
            for _ in range(100):
                actions = self.gridworld.get_actions(state)
                if not actions: break
                action = self.choose_action(state)
                transitions = self.gridworld.get_transitions(state, action)
                next_state = random.choices(list(transitions.keys()), 
                                            weights=list(transitions.values()), k=1)[0]
                reward = self.gridworld.get_reward(state, action, next_state) 
                episode_history.append((state, action, reward))
                state = next_state

            # 2. Process Episode (First-Visit)
            g = 0 
            visited_state_actions = set()
            for i in range(len(episode_history) - 1, -1, -1):
                s, a, r = episode_history[i]
                g = r + self.gamma * g
                if (s, a) not in visited_state_actions:
                    visited_state_actions.add((s, a))
                    self.returns_sum[s][a] += g
                    self.returns_count[s][a] += 1
                    self.q_values[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]

            # 3. Check for Policy Convergence 
            current_policy = {s: self.getPolicy(s) for s in self.gridworld.states if self.gridworld.get_actions(s)}
            
            if current_policy == prev_policy:
                stable_count += 1
            else:
                stable_count = 0
            
            prev_policy = current_policy

            # Mark convergence if stable for 300 episodes
            if stable_count >= 300 and self.conv_episode == self.episodes:
                self.conv_episode = ep - 300

        self.runtime = time.time() - start_time

    def getPolicy(self, state: 'Gridworld.State'):
        actions = self.gridworld.get_actions(state)
        if not actions: return None
        max_val = self.get_max_q(state)
        best_actions = [a for a, v in self.q_values[state].items() if abs(v - max_val) < 1e-9]
        best_actions.sort(key=lambda x: str(x))
        return best_actions[0]

    def getValue(self, state: 'Gridworld.State') -> float:
        actions = self.gridworld.get_actions(state)
        if not actions:
            val = self.gridworld._grid_value(state)
            return float(val) if isinstance(val, (int, float)) else 0.0
        return self.get_max_q(state)

# --- Output Formatting ---

def print_grid_with_axes(agent, grid, label, mode="value"):
    n, m = len(grid), len(grid[0])
    symbols = {Gridworld.Action.Up: "↑", Gridworld.Action.Down: "↓", 
               Gridworld.Action.Left: "←", Gridworld.Action.Right: "→", None: "X"}
    
    title = "Value Function" if mode == "value" else "Policy"
    print(f"\n{'='*55}\n {title} — {label}\n{'='*55}")
    header = "      " + " ".join([f"y={i: <5}" for i in range(m)])
    print(header)
    print("   " + "-" * (len(header) - 3))

    for row_idx in range(n):
        x = n - 1 - row_idx
        row_str = f"x={x} |"
        for y in range(m):
            cell = grid[row_idx][y]
            if cell == '#':
                display = "XXXXXX" if mode == "value" else "  #   "
            elif isinstance(cell, (int, float)):
                display = " GOAL " if mode == "value" else "  G   "
            else:
                if mode == "value":
                    display = f"{agent.getValue((x, y)):+6.2f}"
                else:
                    a = agent.getPolicy((x, y))
                    display = f"  {symbols[a]}   "
            row_str += f" {display}"
        print(row_str)

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
    mc = MonteCarloAgent(game, epsilon=0.2, gamma=0.9, episodes=20000)
    print("Training Monte Carlo Agent...")
    mc.train()
    
    print_grid_with_axes(mc, grid, "Monte Carlo", mode="value")
    print_grid_with_axes(mc, grid, "Monte Carlo", mode="policy")

    print(f"\n{'='*70}")
    print(f"{'Algorithm':<20} | {'Total Episodes':<15} | {'Conv. Episode':<15} | {'Time (s)'}")
    print("-" * 70)
    print(f"{'Monte Carlo':<20} | {mc.episodes:<15} | {mc.conv_episode:<15} | {mc.runtime:.4f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()