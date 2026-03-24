import GUI
import HAL
from ompl import base as ob
from ompl import geometric as og
import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from math import hypot
import time

CELL_SIZE = 25

obstacles = []
goals2D = []
goals3D = [(3.728, 0.579), (3.728, -1.242), (3.728, -3.039), (3.728, -4.827), (3.728, -6.781), (3.728, -8.665)]
dimensions = [0, 0, 415, 279]
path3D = []

state = 0
i = 1

reached_goal = False
nav = True
comeback = False
alligment = False
going = True

def reset_map(orig_map):
    rows, cols, _ = orig_map.shape
    old_map = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            old_map[i, j] = np.average(orig_map[i, j]) * 127

    return old_map

def getPose2D():
    robot_pose_X = HAL.getPose3d().x
    robot_pose_Y = HAL.getPose3d().y

    transformation_Matrix = np.array([[-1, 0, 0, 6.8], [0, -1, 0, 10.31], [0, 0, 1, 0], [0, 0, 0, 1]])

    scale_x = 279 / 13.6
    scale_y = 415 / 20.62

    robot_pose_3d = np.array([robot_pose_X, robot_pose_Y, 0, 1])

    robot_pose_2d = np.dot(transformation_Matrix, robot_pose_3d)

    pose2d_x = robot_pose_2d[0] * scale_x
    pose2d_y = robot_pose_2d[1] * scale_y

    pose2d = (pose2d_x, pose2d_y)

    return pose2d

def getGoals2D():
    for goal in goals3D:
        goal_x = goal[0]
        goal_y = goal[1]

        transformation_matrix = np.array([[-1, 0, 0, 6.8], [0, -1, 0, 10.31], [0, 0, 1, 0], [0, 0, 0, 1]])

        scale_x = 279 / 13.6
        scale_y = 415 / 20.62

        goal_3d = np.array([goal_x, goal_y, 0, 1])
        goal_2d = np.dot(transformation_matrix, goal_3d)

        goal2d_x = goal_2d[0] * scale_x
        goal2d_y = goal_2d[1] * scale_y

        goals2D.append((goal2d_x, goal2d_y))

def getPath3D(path2d):
    for point in path2d:
        x = point[0]
        y = point[1]

        transformation_matrix = np.array([[-1, 0, 0, 6.8], [0, -1, 0, 10.31], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Invert the transformation matrix to get 2D -> 3D
        inverse_matrix = np.linalg.inv(transformation_matrix)

        scale_x = 13.6 / 279
        scale_y = 20.62 / 415

        point_2d = np.array([y * scale_x, x * scale_y, 0, 1])

        point_3d = np.dot(transformation_matrix, point_2d)

        path3D.append((point_3d[0], point_3d[1]))

def expandObstacles(src_map):
    erosion_type = cv.MORPH_RECT
    erosion_size = 2

    element = cv.getStructuringElement(erosion_type, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    erosion_dst = cv.erode(src_map, element)

    return erosion_dst

map_2d = GUI.getMap('/resources/exercises/amazon_warehouse/images/map.png')
clean_map = reset_map(map_2d)
expanded_map = expandObstacles(clean_map)
expanded_map_clean = expanded_map.copy()

# Specify valid state condition
def isStateValid(state):
    global comeback, expanded_map
    x, y, theta = state.getX(), state.getY(), state.getYaw()

    if comeback:
        width, height = 48, 22
    else:
        width, height = 12, 12

    # Center of the rectangle
    cx, cy = int(x), int(y)

    # Coordinates of the rectangle's vertices without rotation (in the local system)
    half_width = width / 2
    half_height = height / 2
    vertices = np.array([
        [-half_width, -half_height],
        [half_width, -half_height],
        [half_width, half_height],
        [-half_width, half_height]
    ])

    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Rotate the vertices around the center
    rotated_vertices = np.dot(vertices, rotation_matrix.T) + [cx, cy]

    # Calculate the bounding box of the rotated rectangle
    x_coords = rotated_vertices[:, 0]
    y_coords = rotated_vertices[:, 1]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    occupied = False
    # Traverse the rows and columns within the bounding box
    for j in range(y_min, y_max):
        for i in range(x_min, x_max):
            # Verify if the point is within bounds and inside the rotated polygon
            if 0 <= i < expanded_map.shape[1] and 0 <= j < expanded_map.shape[0]:
                if cv.pointPolygonTest(rotated_vertices.astype(np.int32), (i, j), False) >= 0:
                    if expanded_map[j, i] <= 105:
                        occupied = True
            else:
                return False

    # Draw the rectangle on the map
    if occupied:
        cv.polylines(expanded_map, [rotated_vertices.astype(np.int32)], isClosed=True, color=128, thickness=1)
    else:
        cv.polylines(expanded_map, [rotated_vertices.astype(np.int32)], isClosed=True, color=131, thickness=1)

    GUI.showNumpy(expanded_map)
    return not occupied

def plan(pose2d, goal):
    # Construct the robot state space in which we're planning. We're
    # planning in [0,1]x[0,1], a subset of R^2.
    space = ob.SE2StateSpace()

    # Set state space's lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, dimensions[0])
    bounds.setLow(1, dimensions[1])
    bounds.setHigh(0, dimensions[2])
    bounds.setHigh(1, dimensions[3])
    space.setBounds(bounds)

    # Construct a space information instance for this state space
    si = ob.SpaceInformation(space)
    # Set state validity checking for this space
    si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    # Set our robot's starting and goal state
    start = ob.State(space)
    start().setX(pose2d[1])
    start().setY(pose2d[0])
    start().setYaw(math.pi / 4)
    goal_state = ob.State(space)

    goal_state().setX(goal[1])
    goal_state().setY(goal[0])
    goal_state().setYaw(math.pi / 4)

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)

    # Set the start and goal states
    pdef.setStartAndGoalStates(start, goal_state)

    # Create a planner for the defined space
    planner = og.RRTstar(si)
    planner.setRange(10)

    # Set the problem we are trying to solve for the planner
    planner.setProblemDefinition(pdef)

    # Perform setup steps for the planner
    planner.setup()

    # Solve the problem and print the solution if it exists
    solved = planner.solve(5.0)
    if solved:
        if draw_path(pdef.getSolutionPath(), dimensions, goal) == -1:
            return -1

        else:
            return pdef.getSolutionPath()

def create_numpy_path(states):
    lines = states.splitlines()
    length = len(lines) - 1
    array = np.zeros((length, 2))

    for i in range(length):
        array[i][0] = float(lines[i].split(" ")[0])
        array[i][1] = float(lines[i].split(" ")[1])
    return array

def draw_path(solution_path, dimensions, goal):
    matrix = solution_path.printAsMatrix()
    path = create_numpy_path(matrix)
    if not isGoalReached(solution_path, goal):
        return -1

    GUI.showPath(path)

# Verifies if the goal is reached
def isGoalReached(solution_path, goal):
    # Verify if the solution_path is valid
    if solution_path is None:
        print("There is no defined path.")
        return False

    # Get the last state of the path
    matrix = solution_path.printAsMatrix()
    path = create_numpy_path(matrix)
    last_state = path[-1]  # Last state of the path

    # Calculate the distance to the goal
    goal_x, goal_y = goal  # Goal must be a tuple (x, y)

    distance = hypot(last_state[0] - goal_y, last_state[1] - goal_x)

    return distance <= 4  # Considered reached if the distance is less than or equal to 4 units

def navigation(solution_path):
    if solution_path is None:
        print("There is no defined path to follow.")
        return

    global state

    # Current position of the robot
    x = HAL.getPose3d().x
    y = HAL.getPose3d().y

    # Current orientation of the robot
    theta = HAL.getPose3d().yaw
    theta = (theta + math.pi) % (2 * math.pi) - math.pi

    goal_x, goal_y = solution_path[0], solution_path[1]

    angle = math.atan2(goal_y - y, goal_x - x)
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    difference = angle - theta

    # Adjust the error to keep it between -pi and pi
    difference = (difference + math.pi) % (2 * math.pi) - math.pi

    # Control constants
    K_angular = 0.5  # Gain for angular control
    K_linear = 0.5   # Gain for linear control

    if state == 0:

        if abs(difference) > 0.01:
            # Angular velocity proportional to the error
            angular_velocity = K_angular * difference
            # Limit the maximum angular velocity
            angular_velocity = max(-0.1, min(angular_velocity, 0.1))
            HAL.setW(angular_velocity)

            return -1

        HAL.setW(0)
        state = 1

    if state == 1:
        distance = math.sqrt(((goal_x - x) ** 2) + ((goal_y - y) ** 2))
        if distance > 0.01:
            # Linear velocity proportional to the distance
            linear_velocity = K_linear * distance
            # Limit the maximum linear velocity
            linear_velocity = max(0.02, min(linear_velocity, 0.08))
            HAL.setV(linear_velocity)

            # Angular velocity proportional to the error
            angular_velocity = K_angular * difference
            # Limit the maximum angular velocity
            angular_velocity = max(-0.1, min(angular_velocity, 0.1))
            HAL.setW(angular_velocity)

            return -1

        HAL.setV(0)
        state = 0

    return 0

def alligmentWithShelves():
    # Rotate until the robot is aligned at pi
    theta = HAL.getPose3d().yaw
    theta = (theta + math.pi) % (2 * math.pi) - math.pi

    diference = math.pi - theta
    diference = (diference + math.pi) % (2 * math.pi) - math.pi

    K_angular = 0.5  # Gain for angular control

    if abs(diference) > 0.01:
        # Angular velocity proportional to the error
        angular_velocity = K_angular * diference
        # Limit the maximum angular velocity
        angular_velocity = max(-0.1, min(angular_velocity, 0.1))
        HAL.setW(angular_velocity)

        return -1

    HAL.setW(0)
    return 0

getGoals2D()
solution_path = plan(getPose2D(), goals2D[0])

while solution_path == -1:
    solution_path = plan(getPose2D(), goals2D[0])

matrix = solution_path.printAsMatrix()
path = create_numpy_path(matrix)
getPath3D(path)  # Convert the path coordinates to 3D.

while True:
    # State machine using match-case
    if nav:
        if navigation(path3D[i]) == 0:
            if i < len(path3D) - 1:
                i += 1

            else:
                i = 1
                nav = False
                alligment = True
                    
                    
    if alligment:
        if alligmentWithShelves() == 0:
            if going:
                HAL.lift()
                comeback = True
                alligment = False
                going = False
                
            else:
                HAL.putdown()
                alligment = False
                time.sleep(100)
                exit(0)

    if comeback:
        offset_x_rect = 48 // 2
        offset_y_rect = 22 // 2
        x_inicio_rect = int(goals2D[0][0] - offset_x_rect)
        x_fin_rec = int(goals2D[0][0] + offset_x_rect)
        y_inicio_rect = int(goals2D[0][1] - offset_y_rect)
        y_fin_rect = int(goals2D[0][1] + offset_y_rect)

        expanded_map = expanded_map_clean.copy()

        # Mark the rectangular region on the map
        cv.rectangle(expanded_map, (y_inicio_rect, x_inicio_rect), (y_fin_rect, x_fin_rec), 127, -1)

        GUI.showNumpy(expanded_map)
        
        solution_path = plan(goals2D[0], (144, 210))

        while solution_path == -1:
            solution_path = plan(goals2D[0], (144, 210))

        matrix = solution_path.printAsMatrix()
        path = create_numpy_path(matrix)
        path3D = []
        getPath3D(path)  # Convert the path coordinates to 3D.

        comeback = False
        nav = True
