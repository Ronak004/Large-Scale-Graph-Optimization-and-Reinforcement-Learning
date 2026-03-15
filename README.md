# SC3000/CZ3005 Lab Assignment 1

This project implements search algorithms for pathfinding on the NYC road network and solves sequential decision-making tasks in a Grid World environment using MDP and Reinforcement Learning.

## Prerequisites

- Python 3.8+
- NYC Instance Files: Ensure `G.json`, `Coord.json`, `Dist.json`, and `Cost.json` are placed in the constrained-graph-search folder.

## Installation & Setup

Follow these steps to set up the local environment and install dependencies:

**1. Create a Virtual Environment:**
```bash
python -m venv venv
```

**2. Activate the Virtual Environment:**

- Windows:
```bash
venv\Scripts\activate
```

- macOS/Linux:
```bash
source venv/bin/activate
```

**3. Install Requirements:**
```bash
pip install -r requirements.txt
```

## Usage

Run the main application using the following command:
```bash
python main.py
```

Upon execution, you will be prompted to select the Part and Task number you wish to run.

## Part 1: NYC Pathfinding

Find the shortest path between node `1` and node `50` with an energy budget of 287,932.

- **Task 1:** Solve a relaxed version without energy constraints.
- **Task 2:** Solve using an uninformed search algorithm (e.g., DFS, BFS, UCS).
- **Task 3:** Solve using an A\* search algorithm with a custom heuristic.

## Part 2: Grid World MDP & RL

Solve a 5×5 grid world problem from start `(0,0)` to goal `(4,4)`.

- **Task 1:** Solve as a Markov Decision Process (MDP) with a known model.
- **Task 2:** Solve as a Reinforcement Learning (RL) problem with an unknown model - Monte Carlo Control.
- **Task 3:** Solve as a Reinforcement Learning (RL) problem with an unknown model - Q-Learning.



#### By Aye Su Nandar Michelle, Thum Mun Kuan and Pahwa Ronak

