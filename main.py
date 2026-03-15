import os
import sys
from pathlib import Path

# Setup Paths
ROOT = Path(__file__).resolve().parent
PART1_DIR = ROOT / "constrained-graph-search"
PART2_DIR = ROOT / "sequential-decision-making-rl"

TASKS = {
    "1": {
        "name": "Part 1 Task 1",
        "directory": PART1_DIR,
        "script": "task1.py",
        "desc": "Shortest path with Dijkstra."
    },
    "2": {
        "name": "Part 1 Task 2",
        "directory": PART1_DIR,
        "script": "task2.py",
        "desc": "Uniform cost search with energy budget."
    },
    "3": {
        "name": "Part 1 Task 3",
        "directory": PART1_DIR,
        "script": "task3.py",
        "desc": "A* search with energy budget."
    },
    "4": {
        "name": "Part 2 Task 1",
        "directory": PART2_DIR,
        "script": "task1.py",
        "desc": "Value/policy iteration variant using GridWorld."
    },
    "5": {
        "name": "Part 2 Task 2",
        "directory": PART2_DIR,
        "script": "task2.py",
        "desc": "Future implementation for Task 2."
    },
    "6": {
        "name": "Part 2 Task 3",
        "directory": PART2_DIR,
        "script": "task3.py",
        "desc": "Q-learning variant using GridWorld."
    },
}

def run_task(choice):
    task = TASKS[choice]
    
    if task["script"] is None:
        print(f"\nTask not yet implemented.")
        input("Press Enter to continue...")
        return

    script_path = task["directory"] / task["script"]
    
    if not script_path.exists():
        print(f"\nError: {task['script']} not found.")
        input("Press Enter to continue...")
        return

    print("=" * 60)
    print(f"RUNNING: {task['name']}")
    print("=" * 60)
    
    os.chdir(task["directory"])
    os.system(f'"{sys.executable}" {task["script"]}')
    
    print("\n" + "=" * 60)
    input("Finished. Press Enter to return to menu...")

def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("                AI PROJECT LAUNCHER")
        print("=" * 60)
        
        current_part = ""
        for num, info in TASKS.items():
            # Section Headers
            part_label = "PART 1: GRAPH SEARCH" if int(num) <= 3 else "PART 2: REINFORCEMENT LEARNING"
            if part_label != current_part:
                print(f"\n--- {part_label} ---")
                current_part = part_label
            
            # Print number, name, and description
            print(f"{num}. {info['name']}")
            print(f"   {info['desc']}")

        print("\nQ. Quit")
        print("=" * 60)
        
        choice = input("Select a number: ").strip().lower()

        if choice == 'q':
            break
        elif choice in TASKS:
            run_task(choice)
        else:
            print("Invalid selection.")
            import time
            time.sleep(0.5)

if __name__ == "__main__":
    main()