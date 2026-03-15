from gridworld2 import *

class ValueIterationAgent:
    def __init__(self, gridworld: Gridworld, discount_factor: float):
        self.gridworld = gridworld
        self.discount_factor = discount_factor
        self.values = {state: 0.0 for state in gridworld.states}

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
        while True:
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
        while True:
            self._policy_evaluation()
            changed = self._policy_improvement()
            if not changed:
                break

    def getPolicy(self, state: Gridworld.State):
        actions = self.gridworld.get_actions(state)
        if not actions:
            return None
        return self.policy[state]

    def getValue(self, state: Gridworld.State) -> float:
        return self.values[state]

def print_value_function(agent, grid, label):
    n, m = len(grid), len(grid[0])
    print(f"\n{'='*55}")
    print(f"  Value Function — {label}")
    print(f"{'='*55}")
    for row_idx in range(n):
        row = ""
        for col_idx in range(m):
            x = n - 1 - row_idx
            y = col_idx
            
            cell = grid[row_idx][col_idx]
            if cell == '#':
                row += "  XXXX "
            elif isinstance(cell, float):
                row += "  GOAL "
            else:
                row += f" {agent.getValue((x, y)):+6.2f}"
        print(f"  {row}")


def print_policy(agent: ValueIterationAgent, grid: tuple, label: str):
    symbols = {
        Gridworld.Action.Up:    "↑",
        Gridworld.Action.Down:  "↓",
        Gridworld.Action.Left:  "←",
        Gridworld.Action.Right: "→",
        None:                   "X",
    }
    n, m = len(grid), len(grid[0])
    print(f"\n{'='*55}")
    print(f"  Policy — {label}")
    print(f"{'='*55}")
    for row_idx in range(n):
        row = ""
        for col_idx in range(m):
            x, y = n - 1 - row_idx, col_idx # Map row/col to x/y
            cell = grid[row_idx][col_idx]
            if cell == '#': row += "  # "
            elif isinstance(cell, float): row += "  G "
            else:
                a = agent.getPolicy((x, y))
                row += f"  {symbols[a]} "
        print(row)



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
    print_value_function(vi, grid, "Value Iteration")
    print_policy(vi, grid, "Value Iteration")

    pi = PolicyIterationAgent(game, discount_factor=0.9)
    pi.iterate()
    print_value_function(pi, grid, "Policy Iteration")
    print_policy(pi, grid, "Policy Iteration")

if __name__ == '__main__':
    main()