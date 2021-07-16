from enum import Enum
from queue import PriorityQueue
import numpy as np

from shapely.geometry import Point
import random
import time
import csv
from shapely.geometry import Polygon
from sklearn.neighbors import KDTree
import networkx as nx
import numpy.linalg as LA
from shapely.geometry import Polygon, Point, LineString
from skimage.draw import *

def create_polygons(data, safety_distance):
    # Valor maximo y minimo de Norte
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # Valor maximo y minimo de Este
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # Calculamos el ancho y alto del grid con
    # los valores maximos y minimos, considerando que
    # discretizamos a una unidad
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    polygons = []
    alt_poly = []
    # Buscamos los polygonos
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
            int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
            int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
            int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
        ]
        x1 = (obstacle[2], obstacle[0])
        x2 = (obstacle[2], obstacle[1])
        y2 = (obstacle[3], obstacle[0])
        y1 = (obstacle[3], obstacle[1])
        coords = [x1, x2, y1, y2]
        poly = Polygon(coords)
        polygons.append(poly)
        alt_poly.append(alt + d_alt)

    return polygons, alt_poly, north_size, east_size

def can_connect(n1, n2, grid):
    rr, cc = line(int(n1[1]),int(n1[0]),int(n2[1]),int(n2[0]))
    for i in range(len(rr)):
        if (grid[rr[i]][cc[i]] == 1):
            return False
    return True

def create_graph(nodes, k, grid):
    g = nx.Graph()
    tree = KDTree(nodes)
    for n1 in nodes:
        idxs = tree.query([n1], k, return_distance=False)[0]

        for idx in idxs:
            n2 = tuple(nodes[idx])
            n1 = tuple(n1)
            if (n2 == n1):
                continue

            if can_connect(n1, n2, grid):
                g.add_edge(n1, n2, weight=1)
    return g

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.
    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def a_star(graph, heuristic, start, goal):
    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost

def heuristic(n1, n2):
    return LA.norm(np.array(n2) - np.array(n1))

def read_global_home(fname):
    import re
    first_line = open(fname).readline()
    print(first_line)
    coord = re.match(r'^lat0 (.*), lon0 (.*)$', first_line)
    if coord:
        lat = coord.group(1)
        lon = coord.group(2)
    return float(lat), float(lon)

def prune_path(path, epsilon=1e-6):
    
    def point(p):
        return np.array([p[0], p[1], 1.]).reshape(1, -1)

    def collinearity_check(p1, p2, p3):   
        m = np.concatenate((p1, p2, p3), 0)
        det = np.linalg.det(m)
        return abs(det) < epsilon

    pruned_path = [p for p in path]
    i = 0
    while i < len(pruned_path) - 2:
        p1 = point(pruned_path[i])
        p2 = point(pruned_path[i+1])
        p3 = point(pruned_path[i+2])
        collinear = collinearity_check(p1, p2, p3)
        if collinear:
            pruned_path.remove(pruned_path[i+1])
        else:
            i += 1
    return pruned_path