import numpy as np
import heapq

import pygame as pg

from Config.settings import *

vec = pg.math.Vector2

WALL_OFFSET = 0

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def convert_from_game_loc(game_location):
    map_ind = game_location // TILESIZE
    array_location = (int(map_ind[1]), int(map_ind[0]))
    return array_location

def convert_to_game_loc(array_location, map):
    offset = find_offset(array_location, map)
    game_loc_corner = vec(array_location[1], array_location[0]) * TILESIZE
    game_loc_center = game_loc_corner + vec(offset[1], offset[0])
    return game_loc_center

def find_offset(array_location, map):
    left_dirs = [(0,-1),(-1,-1),(1,-1)]
    right_dirs = [(0,1),(-1,1),(1,1)]
    wall_to_left = False
    wall_to_right = False
    for i, j in left_dirs:
        if map[(array_location[0] + i, array_location[1] + j)] == 1:
            wall_to_left = True
    for i, j in right_dirs:
        if map[(array_location[0] + i, array_location[1] + j)] == 1:
            wall_to_right = True
    if wall_to_left and not wall_to_right:
        LR_offset = (1 + WALL_OFFSET) * TILESIZE
    elif not wall_to_left and wall_to_right:
        LR_offset = - WALL_OFFSET * TILESIZE
    else:
        LR_offset = TILESIZE / 2

    above_dirs = [(-1,-1),(-1,0),(-1,1)]
    bellow_dirs = [(1,-1),(1,0),(1,1)]
    wall_above = False
    wall_below = False
    for i, j in above_dirs:
        if map[(array_location[0] + i, array_location[1] + j)] == 1:
            wall_above = True
    for i, j in bellow_dirs:
        if map[(array_location[0] + i, array_location[1] + j)] == 1:
            wall_below = True
    if wall_above and not wall_below:
        UD_offset = (1 + WALL_OFFSET) * TILESIZE
    elif not wall_above and wall_below:
        UD_offset = - WALL_OFFSET * TILESIZE
    else:
        UD_offset = TILESIZE / 2

    offset = (UD_offset, LR_offset)
    return offset

def convert_to_map_route(start_location, array_route, map):
    map_route = []
    start_waypoint = convert_to_game_loc(start_location, map)
    for waypoint in array_route:
        map_waypoint = convert_to_game_loc(waypoint, map)
        # lin_interp_pts = np.linspace(1/10,1,2)
        # for interp_loc in lin_interp_pts:
        #     new_wp = pg.math.Vector2.lerp(start_waypoint, map_waypoint, interp_loc)
        #     map_route.append(new_wp)
        # start_waypoint = map_waypoint
        map_route.append(map_waypoint)
    return map_route

class A_Star():
    def __init__(self, map_raw):
        self.map = self._refine_map(map_raw)
        self.col_edge = (0, len(self.map[0]) - 1)
        self.row_edge = (0, len(self.map[:,0])-1)
        self.max_counter = (self.col_edge[1] + 1) * (self.row_edge[1] + 1)
        self.neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    
    def _refine_map(self, map_raw):
        new_map = []
        for line in map_raw:
            line = line.strip('\n')
            row = []
            for element in line:
                row.append(element)
            new_map.append(row)
        new_map = np.asarray(new_map)
        refined_map = np.empty_like(new_map, dtype = int)
        for i, row in enumerate(refined_map):
            for j, col in enumerate(row):
                if new_map[i][j] == '1':
                    refined_map[i][j] = 1
                else:
                    refined_map[i][j] = 0
        return refined_map
 
    def astar(self, start, goal):
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                return data
            close_set.add(current)

            for i, j in self.neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + heuristic(current, neighbor)
                if 0 <= neighbor[0] < self.map.shape[0]:
                    if 0 <= neighbor[1] < self.map.shape[1]:                
                        if self.map[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        # array bound y walls
                        continue
                else:
                    # array bound x walls
                    continue
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return None

    def find_route(self, start, goal):
        start_location = convert_from_game_loc(start)
        goal_location = convert_from_game_loc(goal)
        route = self.astar(start_location, goal_location)
        #route = route + [start_location]
        route = route[::-1]
        map_route = convert_to_map_route(start_location, route, self.map)
        return map_route

