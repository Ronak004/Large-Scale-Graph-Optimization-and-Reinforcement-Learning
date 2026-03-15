# Import relevant libraries
import heapq
import math

# Load JSON files
from settings import load_settings
coord, cost, dist, g = load_settings()

# Defining the heuristic used as the Euclidean distance
def euclidean_heuristic(node, goal, coord):
    x1, y1 = coord[node]
    x2, y2 = coord[goal]
    return math.hypot(x1 - x2, y1 - y2)

# Builds a reversed version of the directed graph (every edge u->v becomes v->u)
# Used to run Dijkstra backwards from the goal to pre-compute minimum energy costs
def build_reverse_graph(graph):
    reverse_graph = {}

    for u, neighbors in graph.items():
        for v in neighbors:
            if v not in reverse_graph:
                reverse_graph[v] = []
            reverse_graph[v].append(u)

    return reverse_graph

# Runs Dijkstra on the reversed graph starting from the goal node
# Returns the minimum energy cost needed to reach the goal from every node
def reverse_dijkstra_min_energy(goal, reverse_graph, cost):
    min_energy = {goal: 0}
    pq = [(0, goal)]

    while pq:
        curr_energy, node = heapq.heappop(pq)

        if curr_energy != min_energy[node]:
            continue

        for pred in reverse_graph.get(node, []):
            edge_key = f"{pred},{node}"
            new_energy = curr_energy + cost[edge_key]

            if pred not in min_energy or new_energy < min_energy[pred]:
                min_energy[pred] = new_energy
                heapq.heappush(pq, (new_energy, pred))

    return min_energy

# Returns True if the new (accumulated_distance, accumulated_energy) label is dominated by any existing label at a node
# A label is dominated when there is already one that is no worse in both distance AND energy
def is_dominated(state_list, new_dist, new_cost):
    for old_cost, old_dist, _ in state_list:
        if old_cost <= new_cost and old_dist <= new_dist:
            return True
    return False

# Removes any existing labels at a node that are now dominated by the new (accumulated_distance, accumulated_energy) label
def remove_dominated(state_list, new_dist, new_cost):
    filtered = []

    for old_cost, old_dist, sid in state_list:
        if not (new_cost <= old_cost and new_dist <= old_dist):
            filtered.append((old_cost, old_dist, sid))

    return filtered

# Walks back through parent pointers from the goal state to the start node, then reverses the result
def reconstruct_path(parent, state_info, goal_sid):
    path = []
    sid = goal_sid

    while sid is not None:
        node, _, _ = state_info[sid]
        path.append(node)
        sid = parent[sid]

    path.reverse()
    return path

# Implementation of multi-label A* search, taking into consideration the energy budget
# Each node holds (energy, distance) labels to track both simultaneously
def astar_with_energy_budget(graph, dist, cost, coord, start, goal, budget):
    reverse_graph = build_reverse_graph(graph)
    min_energy_to_goal = reverse_dijkstra_min_energy(goal, reverse_graph, cost)

    # If even the minimum energy needed from start exceeds budget, no solution exists
    if start not in min_energy_to_goal or min_energy_to_goal[start] > budget:
        return None

    pq = []
    parent = {}
    state_info = {}
    label_sets = {}   # node -> list of (energy, distance, sid)

    sid_counter = 0
    start_sid = sid_counter
    sid_counter += 1

    start_g = 0.0
    start_c = 0.0
    start_h = euclidean_heuristic(start, goal, coord)

    # Initialise the priority queue with the starting node, prioritised by f = accumulated_distance + heuristic
    heapq.heappush(pq, (start_g + start_h, start_g, start_c, start, start_sid))
    parent[start_sid] = None
    state_info[start_sid] = (start, start_g, start_c)
    label_sets[start] = [(start_c, start_g, start_sid)]

    expanded_states = 0

    while pq:
        f, curr_dist, curr_cost, u, sid = heapq.heappop(pq)

        # Skip this state if it has since been dominated and removed from the label set
        valid = False
        for saved_cost, saved_dist, saved_sid in label_sets.get(u, []):
            if saved_sid == sid and saved_cost == curr_cost and saved_dist == curr_dist:
                valid = True
                break

        if not valid:
            continue

        expanded_states += 1

        # If the current node is the goal, reconstruct and return the path
        if u == goal:
            path = reconstruct_path(parent, state_info, sid)
            return {
                "path": path,
                "distance": curr_dist,
                "energy": curr_cost,
                "expanded_states": expanded_states
            }

        # Explore the current node's neighbours
        for v in graph[u]:
            edge_key = f"{u},{v}"

            new_dist = curr_dist + dist[edge_key]
            new_cost = curr_cost + cost[edge_key]

            # Discard if this edge alone pushes energy over the budget
            if new_cost > budget:
                continue

            # Discard if the minimum remaining energy to goal would still exceed the budget
            if v not in min_energy_to_goal:
                continue
            if new_cost + min_energy_to_goal[v] > budget:
                continue

            v_labels = label_sets.get(v, [])

            # Skip if an existing label already dominates this new one
            if is_dominated(v_labels, new_dist, new_cost):
                continue

            # Remove old labels dominated by the new one
            v_labels = remove_dominated(v_labels, new_dist, new_cost)

            # Register the new state, add it to the label set, and push it onto the queue
            new_sid = sid_counter
            sid_counter += 1

            v_labels.append((new_cost, new_dist, new_sid))
            label_sets[v] = v_labels

            parent[new_sid] = sid
            state_info[new_sid] = (v, new_dist, new_cost)

            h = euclidean_heuristic(v, goal, coord)
            heapq.heappush(pq, (new_dist + h, new_dist, new_cost, v, new_sid))

    return None

start_node = "1"
goal_node = "50"
energy_budget = 287932

result = astar_with_energy_budget(
    graph=g,
    dist=dist,
    cost=cost,
    coord=coord,
    start=start_node,
    goal=goal_node,
    budget=energy_budget
)

if result is None:
    print("No feasible path found within the energy budget.")
else:
    print("Shortest path:", "->".join(result["path"]))
    print("Shortest distance:", result["distance"])
    print("Total energy cost:", result["energy"])