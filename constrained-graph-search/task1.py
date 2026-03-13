import heapq
from settings import load_settings

def dijkstra(start_node, end_node, graph, dist_map, cost_map):
    # priority_queue stores (cumulative_distance, current_node)
    pq = [(0, start_node)]

    # distances[node] stores the minimum distance from start to node
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    # parent[node] stores the predecessor to reconstruct the path
    parent = {start_node: None}

    while pq:
        current_dist, u = heapq.heappop(pq)

        # If we reached the target, we can stop early
        if u == end_node:
            break

        # Standard Dijkstra: if we found a longer path, skip
        if current_dist > distances[u]:
            continue

        # Explore neighbors
        for v in graph.get(u, []):
            weight = dist_map.get(f"{u},{v}", float('inf'))
            new_dist = current_dist + weight

            if new_dist < distances[v]:
                distances[v] = new_dist
                parent[v] = u
                heapq.heappush(pq, (new_dist, v))

    # Reconstruct the path from end_node to start_node
    path = []
    curr = end_node
    total_energy = 0

    if distances[end_node] == float('inf'):
        return None, float('inf'), 0

    while curr is not None:
        path.append(curr)
        prev = parent[curr]
        if prev is not None:
            # Calculate energy cost along the found shortest path
            total_energy += cost_map.get(f"{prev},{curr}", 0)
        curr = prev

    path.reverse()
    return path, distances[end_node], total_energy

def main():
    # Define start and end nodes as per Task 1 requirements
    start_node = '1'
    end_node = '50'

    print("Loading data files...")
    coord, cost, dist, g = load_settings()

    print(f"Solving Task 1: Shortest path from {start_node} to {end_node}...")
    path, total_dist, total_energy = dijkstra(start_node, end_node, g, dist, cost)

    if path:
        # Format output as required: Shortest path: 1->...->50
        path_str = "->".join(path)
        print(f"Shortest path: {path_str}")
        print(f"Shortest distance: {total_dist}")
        print(f"Total energy cost: {total_energy}")
    else:
        print("No path found.")

if __name__ == "__main__":
    main()