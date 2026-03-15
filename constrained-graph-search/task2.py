# Import relevant libraries
import heapq

# Load JSON files
from settings import load_settings
coord, cost, dist, g = load_settings()

# Implementation of a priority queue using a minimising heap
class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, priority, item):
        entry = (priority, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (priority, item) = heapq.heappop(self.heap)
        return priority, item

    def is_empty(self):
        return len(self.heap) == 0

# Implementation of Uniform Cost Search, taking into consideration the energy budget
def uniform_cost_search(graph, start, goal, energy_budget):
    # Initialize the priority queue with the starting node
    priority_queue = PriorityQueue()
    priority_queue.push(0, (start, 0, None))

    # Keep track of visited nodes with format node : (accumulated_distance, accumulated_energy, parent_node)
    # Nodes are added when first discovered, and updated if a shorter path is found using the node
    visited = {start: (0, 0, None)}

    while not priority_queue.is_empty():
        # Pop the node with the lowest accumulated distance from the priority queue
        accumulated_distance, (node, accumulated_energy, parent) = priority_queue.pop()

        # If the current node is the goal, reconstruct and return the path
        if node == goal:
            return reconstruct_path(visited, node)

        # Explore the current node's neighbours
        for neighbour in graph[node]:
            distance_to_neighbour = accumulated_distance + dist[f"{node},{neighbour}"]
            energy_to_neighbour = accumulated_energy + cost[f"{node},{neighbour}"]

            # Add the neighbour to the priority queue if its addition to the path would not exceed the energy budget
            # and if we haven't already found a shorter path to this neighbour
            if energy_to_neighbour <= energy_budget:
                if neighbour not in visited or distance_to_neighbour < visited[neighbour][0]:
                    visited[neighbour] = (distance_to_neighbour, energy_to_neighbour, node)
                    priority_queue.push(distance_to_neighbour, (neighbour, energy_to_neighbour, node))

    return None  # If no path is found


def reconstruct_path(visited, goal_node):
    # Walk backwards from the goal node to the start node using parent pointers
    path = []
    node = goal_node
    while node is not None:
        accumulated_distance, accumulated_energy, parent = visited[node]
        path.append(node)  # Append the node name
        node = parent
    path.reverse()

    total_dist = visited[goal_node][0]
    total_energy = visited[goal_node][1]
    return path, total_dist, total_energy


if __name__ == "__main__":
    result = uniform_cost_search(g, "1", "50", 287932)
    if result:
        path, total_dist, total_energy = result
        print(f"Shortest path: {'->'.join(path)}.")
        print(f"Shortest distance: {total_dist}.")
        print(f"Total energy cost: {total_energy}.")
    else:
        print("No path found within energy budget.")