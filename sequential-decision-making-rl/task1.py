import time

from gridworld import *

class ValueIterationAgent:
    def __init__(self, gridworld: Gridworld, discount_factor: float):
        self.gridworld = gridworld
        self.discount_factor = discount_factor
        self.values = {state: 0.0 for state in gridworld.states}
        self.iterations = 0 
        self.runtime = 0.0

    def calc_q_value(self, state: Gridworld.State, action: Gridworld.Action) -> float:
        q_value = 0.0
        transitions = self.gridworld.get_transitions(state, action)
        for next_state, probability in transitions.items():
            q_value += probability * (
                self.gridworld.get_reward(state, action, next_state)
                + self.discount_factor * self.values.get(next_state, 0.0)
            )
        return q_value

    def iterate(self):
        start_time = time.time()
        while True:
            self.iterations += 1
            new_values = {}
            delta = 0.0
            for state in self.gridworld.states:
                actions = self.gridworld.get_actions(state)
                if actions:
                    new_values[state] = max(
                        self.calc_q_value(state, action) for action in actions
                    )
                else:
                    new_values[state] = 0.0
                delta = max(delta, abs(new_values[state] - self.values.get(state, 0.0)))
            self.values = new_values
            if delta < 1e-6:
                break

        self.runtime = time.time() - start_time

    def getPolicy(self, state: Gridworld.State):
        actions = self.gridworld.get_actions(state)
        if not actions:
            return None
        return max(actions, key=lambda a: self.calc_q_value(state, a))

    def getValue(self, state: Gridworld.State) -> float:
        return self.values[state]

class PolicyIterationAgent(ValueIterationAgent):
    def __init__(self, gridworld: Gridworld, discount_factor: float):
        super().__init__(gridworld, discount_factor)
        # Initial policy: always go Right
        self.policy = {}
        for state in gridworld.states:
            actions = gridworld.get_actions(state)
            self.policy[state] = Gridworld.Action.Right if actions else None

    def _policy_evaluation(self):
        while True:
            new_values = {}
            delta = 0.0
            for state in self.gridworld.states:
                action = self.policy[state]
                if action is None:
                    new_values[state] = self.values[state]
                else:
                    new_values[state] = self.calc_q_value(state, action)
                delta = max(delta, abs(new_values[state] - self.values.get(state, 0.0)))
            self.values = new_values
            if delta < 1e-6:
                break

    def _policy_improvement(self) -> bool:
        policy_changed = False
        for state in self.gridworld.states:
            actions = self.gridworld.get_actions(state)
            if not actions:
                continue
            best_action = max(actions, key=lambda a: self.calc_q_value(state, a))
            if self.calc_q_value(state, best_action) > self.calc_q_value(state, self.policy[state]):
                self.policy[state] = best_action
                policy_changed = True
        return policy_changed

    def iterate(self):
        start_time = time.time()
        while True:
            self.iterations += 1
            self._policy_evaluation()
            changed = self._policy_improvement()
            if not changed:
                break
        self.runtime = time.time() - start_time

    def getPolicy(self, state: Gridworld.State):
        actions = self.gridworld.get_actions(state)
        if not actions:
            return None
        return self.policy[state]

    def getValue(self, state: Gridworld.State) -> float:
        return self.values[state]

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
        (' ', '#', ' ', '#', ' ',),
        (' ', ' ', ' ', ' ', ' '),
        ('S', ' ', ' ', ' ', ' '),
    )
    game = Gridworld(noise=0.2, living_reward=-1, grid=grid)

    vi = ValueIterationAgent(game, discount_factor=0.9)
    vi.iterate()
    print_grid_with_axes(vi, grid, "Value Iteration", mode="value")
    print_grid_with_axes(vi, grid, "Value Iteration", mode="policy")

    pi = PolicyIterationAgent(game, discount_factor=0.9)
    pi.iterate()
    print_grid_with_axes(pi, grid, "Policy Iteration", mode="value")
    print_grid_with_axes(pi, grid, "Policy Iteration", mode="policy")

    print(f"\n{'='*65}")
    print(f"{'Algorithm':<20} {'Episodes':<12} {'Conv. Iteration':<18} {'Time (s)':<10}")
    print("-" * 65)
    print(f"{'Value Iteration':<20} {'N/A':<12} {vi.iterations:<18} {vi.runtime:.5f}")
    print(f"{'Policy Iteration':<20} {'N/A':<12} {pi.iterations:<18} {pi.runtime:.5f}")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()