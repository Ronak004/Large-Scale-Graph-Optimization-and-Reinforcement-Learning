import json

def load_settings():
    with open('Coord.json', 'r') as f:
        coord = json.load(f)
    with open('Cost.json', 'r') as f:
        cost = json.load(f)
    with open('Dist.json', 'r') as f:
        dist = json.load(f)
    with open('G.json', 'r') as f:
        g = json.load(f)
    return coord, cost, dist, g