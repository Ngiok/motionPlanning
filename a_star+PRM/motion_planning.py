import argparse
import time
import msgpack
from enum import Enum, auto
from queue import PriorityQueue
import numpy as np

from planning_utils import a_star, heuristic, create_grid, read_global_home, prune_path, create_polygons, create_graph
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


from shapely.geometry import Point
import random
import time
import csv

class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 1

        self.target_position[2] = TARGET_ALTITUDE

        lat0, lon0 = read_global_home('colliders.csv')
        
        self.set_home_position(lon0, lat0, 0)
        
        print('global position {}', format(self.global_position))
 
        local_north, local_east, _ =global_to_local(self.global_position, self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        
        start_time = time.time()
        
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
  
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        
        grid_start_north = int(np.ceil(local_north - north_offset))
        grid_start_east = int(np.ceil(local_east - east_offset))
        grid_start = (grid_start_north, grid_start_east)
        print(grid_start)

        lat_goal, long_goal = 37.793396, -122.398581
        goal_global_position = np.array([long_goal, lat_goal, 0.0 ])
        
        local_north_goal, local_east_goal, _ = global_to_local(goal_global_position, self.global_home)
        
        grid_goal_north = int(np.ceil(local_north_goal - north_offset))
        grid_goal_east = int(np.ceil(local_east_goal - east_offset))
        grid_goal = (grid_goal_north, grid_goal_east)
        
        goal=[grid_goal_east, grid_goal_north]
        start= [ grid_start_east, grid_start_north]
        print('Local Start and Goal: ', grid_start, grid_goal)
        print(grid[grid_goal_north][grid_goal_east])
        
        #PRM
        
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 2
        
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        # Generamos puntos random
        N_PUNTOS = 500
        puntos = []
        puntos_generados=0
        while puntos_generados < N_PUNTOS:
            x=random.randint(0,(len(grid[0])-1))
            y=random.randint(0,(len(grid[0])-1))
            if grid[y,x] == 0:
                puntos.append(Point(x,y))
                puntos_generados+=1
    
        puntos_filtrados = puntos
        
        puntos_filtrados_list = []
        for i in puntos_filtrados:
            puntos_filtrados_list.append([i.x,i.y])
        puntos_filtrados_list.append(start)
        puntos_filtrados_list.append(goal)
        
        grafo = create_graph(np.asarray(puntos_filtrados_list), 10, grid)
       
        start = tuple(start)
        goal = tuple(goal)
        path, cost = a_star(grafo, heuristic, start, goal)
        if not path:
            print('**********************')
            print('Failed to find a path!')
            print('**********************')
        
        path.append(list(goal))
        
        final_time=time.time()
        
        delay=final_time-start_time
        print('delay = ', delay)
        
        
        waypoints = [[int(p[1]) + north_offset, int(p[0]) + east_offset, TARGET_ALTITUDE, 0] for p in path]
        self.waypoints = waypoints
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()