import GUI
import HAL
import numpy as np
import math
from collections import deque

# Definir las dimensiones de la cuadrícula
CELL_SIZE = 34
MAP_LENGTH = 1012
MAP_WIDTH = 1012

celdas_limpias = []
return_cells = []
actualCell = None
path = []

FORWARD = 0
TURN = 1
state = FORWARD
last_pose = None

last_dir = "north"
north = 0
south = math.pi
east = math.pi / 2
west = -math.pi / 2

map_2d = GUI.getMap("/resources/exercises/vacuum_cleaner_loc/images/mapgrannyannie.png")

# Variables adicionales para BFS
BFS_cells = []
BFS_obstacle_cells = []

def parse_laser_data(laser_data):
    laser_polar = []
    laser_xy = []
    for i in range(180):
        dist = laser_data.values[i]
        angle = math.radians(i - 90)
        laser_polar += [(dist, angle)]
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        laser_xy += [(x, y)]
    return laser_polar, laser_xy

def showMapGrid(grid):
    for x in range(map_2d.shape[0]):
        for y in range(map_2d.shape[1]):
            if (map_2d[x][y] != 0).all():
                grid[x, y] = 127

    for y in range(0, 1024, CELL_SIZE):
        grid[y : y + 1, :] = 128

    for x in range(0, 1024, CELL_SIZE):
        grid[:, x : x + 1] = 128

    return grid

def getPose2D():
    robot_pose_X = HAL.getPose3d().x
    robot_pose_Y = HAL.getPose3d().y
    robot_pose_Z = HAL.getPose3d().z

    transformation_Matrix = np.array([
        [math.cos(-math.pi / 2), -math.sin(-math.pi / 2), 0, 4.07],
        [math.sin(-math.pi / 2),  math.cos(-math.pi / 2), 0, 5.65],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    scale = 101.7
    robot_pose_3d = np.array([robot_pose_X, robot_pose_Y, robot_pose_Z, 1])
    robot_pose_2d = np.dot(transformation_Matrix, robot_pose_3d)
    pose2d = robot_pose_2d * scale
    return (round(pose2d[0]), round(pose2d[1]))

def checkFreeCell(x, y, map_grid):
    xMin = max(0, x - CELL_SIZE // 2)
    yMin = max(0, y - CELL_SIZE // 2)
    xMax = min(MAP_LENGTH, x + CELL_SIZE // 2)
    yMax = min(MAP_LENGTH, y + CELL_SIZE // 2)
    cell = map_grid[xMin:xMax, yMin:yMax]
    return np.sum(cell == 0) == 0

def getFreeCellsCenters(grid):
    centers = []
    for i in range(CELL_SIZE // 2, grid.shape[0], CELL_SIZE):
        for j in range(CELL_SIZE // 2, grid.shape[1], CELL_SIZE):
            if checkFreeCell(i, j, grid):
                centers.append((i, j))
    return centers

def getFrontCoord(free_cell_centers):
    global last_dir, actualCell
    if actualCell is None:
        return (False, actualCell)

    if last_dir == "north":
        next_cell = (actualCell[0] + CELL_SIZE, actualCell[1])
    elif last_dir == "south":
        next_cell = (actualCell[0] - CELL_SIZE, actualCell[1])
    elif last_dir == "east":
        next_cell = (actualCell[0], actualCell[1] + CELL_SIZE)
    elif last_dir == "west":
        next_cell = (actualCell[0], actualCell[1] - CELL_SIZE)

    if next_cell not in free_cell_centers:
        return (False, actualCell)

    return (True, next_cell)

def changeDirection():
    global last_dir
    dirs = ["north", "east", "south", "west"]
    idx = dirs.index(last_dir)
    last_dir = dirs[(idx + 1) % 4]

def getAdjacentCells(cell):
    return [
        (cell[0] + CELL_SIZE, cell[1]),
        (cell[0] - CELL_SIZE, cell[1]),
        (cell[0], cell[1] + CELL_SIZE),
        (cell[0], cell[1] - CELL_SIZE)
    ]

def cellBlocked(cell, free_cell_centers):
    for adj_cell in getAdjacentCells(cell):
        if adj_cell in free_cell_centers:
            return False
    return True

def drawPath(path):
    for x, y in path:
        xMin = max(0, x - CELL_SIZE // 2)
        yMin = max(0, y - CELL_SIZE // 2)
        xMax = min(MAP_LENGTH, x + CELL_SIZE // 2)
        yMax = min(MAP_LENGTH, y + CELL_SIZE // 2)
        grid[xMin:xMax, yMin:yMax] = 128

def drawReturnCells(cells):
    for x, y in cells:
        xMin = max(0, x - CELL_SIZE // 2)
        yMin = max(0, y - CELL_SIZE // 2)
        xMax = min(MAP_LENGTH, x + CELL_SIZE // 2)
        yMax = min(MAP_LENGTH, y + CELL_SIZE // 2)
        grid[xMin:xMax, yMin:yMax] = 129

def drawFreeCells(cells):
    for x, y in cells:
        xMin = max(0, x - CELL_SIZE // 2)
        yMin = max(0, y - CELL_SIZE // 2)
        xMax = min(MAP_LENGTH, x + CELL_SIZE // 2)
        yMax = min(MAP_LENGTH, y + CELL_SIZE // 2)
        grid[xMin:xMax, yMin:yMax] = 132

def updateReturnCells(cell, free_cell_centers, return_cells_list, path):
    for adj_cell in getAdjacentCells(cell):
        if adj_cell in free_cell_centers and adj_cell not in path and adj_cell not in return_cells_list:
            return_cells_list.append(adj_cell)
    if cell in return_cells_list:
        return_cells_list.remove(cell)
    return return_cells_list

def findNearestReturnCell(cell, return_cells_list):
    if not return_cells_list:
        return None
    return min(return_cells_list, key=lambda c: math.dist(cell, c))

def findNearestFreeCell(position, free_cells):
    if not free_cells:
        return None
    return min(free_cells, key=lambda c: math.dist(position, c))

def createMapGrid():
    global grid
    grid = np.zeros((map_2d.shape[0], map_2d.shape[1]))
    showMapGrid(grid)

def initializeRobotCell():
    global actualCell, path, free_cell_centers
    robot_pose = getPose2D()
    nearest_cell = findNearestFreeCell(robot_pose, free_cell_centers)
    if nearest_cell:
        actualCell = nearest_cell
        path.append(actualCell)
        if actualCell in free_cell_centers:
            free_cell_centers.remove(actualCell)
        print("Empezando en celda:", actualCell)

# Funciones BFS
def get_neighbors(cell):
    directions = [(0, CELL_SIZE), (0, -CELL_SIZE), (CELL_SIZE, 0), (-CELL_SIZE, 0)]  # Norte, Sur, Este, Oeste
    neighbors = []
    for d in directions:
        neighbor = (cell[0] + d[0], cell[1] + d[1])
        if neighbor in BFS_cells and neighbor not in BFS_obstacle_cells:
            neighbors.append(neighbor)
    return neighbors

def reconstruct_path(came_from, current):
    path_cells = []
    while current:
        path_cells.append(current)
        current = came_from[current]
    return path_cells[::-1]  # Invertimos para obtener el orden correcto

def bfs(start, goal):
    queue = deque([start])
    came_from = {start: None}
    
    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(came_from, current)
        
        neighbors = get_neighbors(current)

        for neighbor in neighbors:
            if neighbor not in came_from: 
                queue.append(neighbor)
                came_from[neighbor] = current
                
    return None

def drawBFSPath(path):
    for x, y in path:
        xMin = max(0, x - CELL_SIZE // 2)
        yMin = max(0, y - CELL_SIZE // 2)
        xMax = min(MAP_LENGTH, x + CELL_SIZE // 2)
        yMax = min(MAP_LENGTH, y + CELL_SIZE // 2)
        grid[xMin:xMax, yMin:yMax] = 134

def buildPath():
    global actualCell, free_cell_centers, return_cells, path, BFS_cells, BFS_obstacle_cells
    
    # Inicializamos BFS_cells con todas las celdas libres (incluyendo las ya visitadas)
    BFS_cells = free_cell_centers.copy()
    for c in path:
        if c != actualCell:
            BFS_cells.append(c)

    BFS_obstacle_cells = []  # Inicialmente vacío
    
    while free_cell_centers:
        if cellBlocked(actualCell, free_cell_centers):
            print("CELDA BLOQUEADA, BUSCANDO RUTA DE RETORNO")
            nearest_return = findNearestReturnCell(actualCell, return_cells)
            if nearest_return:
                print("CELDA DE RETORNO MÁS CERCANA:", nearest_return)
                
                # Actualizar celdas BFS para la búsqueda
                BFS_cells = free_cell_centers.copy()
                for c in path:
                    if c != actualCell:
                        BFS_cells.append(c)

                BFS_obstacle_cells = []
                
                # Usar BFS para encontrar camino hasta la celda de retorno
                bfs_path = bfs(actualCell, nearest_return)
                if bfs_path and len(bfs_path) > 1:
                    drawBFSPath(bfs_path)
                    GUI.showNumpy(grid)
                    
                    for cell in bfs_path[1:]:
                        path.append(cell)
                        if cell in free_cell_centers:
                            free_cell_centers.remove(cell)
                            return_cells = updateReturnCells(cell, free_cell_centers, return_cells, path)
                    
                    actualCell = bfs_path[-1]
                else:
                    actualCell = nearest_return
                    path.append(actualCell)
                    if actualCell in free_cell_centers:
                        free_cell_centers.remove(actualCell)
                    return_cells = updateReturnCells(actualCell, free_cell_centers, return_cells, path)
            else:
                print("NO HAY CELDAS DE RETORNO DISPONIBLES")
                break
        else:
            success, nextCell = getFrontCoord(free_cell_centers)
            if success:
                actualCell = nextCell
                path.append(actualCell)
                free_cell_centers.remove(actualCell)
                return_cells = updateReturnCells(actualCell, free_cell_centers, return_cells, path)
            else:
                changeDirection()
        
        drawPath(path)
        drawReturnCells(return_cells)
        GUI.showNumpy(grid)

def initMapAndPath():
    global free_cell_centers
    createMapGrid()
    free_cell_centers = getFreeCellsCenters(grid)
    drawFreeCells(free_cell_centers)
    initializeRobotCell()
    buildPath()

def getYaw():
    yaw = HAL.getPose3d().yaw
    if yaw < 0:
        yaw = math.pi + (math.pi - abs(yaw))
    return yaw

def getDirectionAngle(from_cell, to_cell):
    dx = to_cell[0] - from_cell[0]
    dy = to_cell[1] - from_cell[1]
    angle = math.atan2(dy, dx)
    angle += math.pi / 2
    if angle < 0:
        angle += 2 * math.pi
    return angle

def computeOffsets(from_cell, to_cell, current_pos):
    if from_cell[0] != to_cell[0]:  # Movimiento vertical
        fwd_offset = abs(current_pos[0] - to_cell[0])
        lat_offset = to_cell[1] - current_pos[1]
    elif from_cell[1] != to_cell[1]:  # Movimiento horizontal
        fwd_offset = abs(current_pos[1] - to_cell[1])
        lat_offset = to_cell[0] - current_pos[0]
    else:
        fwd_offset = lat_offset = 0
    return fwd_offset, lat_offset

def PID_V(error, Kp=0.015):
    return Kp * error

def PID_W(target_angle, current_yaw, Kp=0.75):
    error = target_angle - current_yaw

    # Normalizar a [-pi, pi]
    if error > math.pi:
        error -= 2 * math.pi
    elif error < -math.pi:
        error += 2 * math.pi
    return Kp * error

def paint_pose(pose, grid, color=131):
    x, y = pose

    x_min = max(0, x - CELL_SIZE // 2)
    x_max = min(grid.shape[0], x + CELL_SIZE // 2)
    y_min = max(0, y - CELL_SIZE // 2)
    y_max = min(grid.shape[1], y + CELL_SIZE // 2)

    grid[x_min:x_max, y_min:y_max] = color

    return grid


# Ejecución principal
initMapAndPath()

state = FORWARD
last_pose = path[0]
path = path[1:]
if path:
    direction = getDirectionAngle(last_pose, path[0]) 
else:
    direction = 0

while True:
    pose2d = getPose2D()
    yaw = getYaw()
    grid = paint_pose(pose2d, grid, 131)

    if state == FORWARD:
        if not path:
            print("Path terminado.")
            HAL.setV(0)
            HAL.setW(0)
            break

        next_cell = path[0]
        fwd_offset, lat_offset = computeOffsets(last_pose, next_cell, pose2d)

        if fwd_offset < 10:
            last_pose = next_cell
            path.pop(0)
            if path:
                print("Siguiente celda objetivo:", path[0])
                direction = getDirectionAngle(last_pose, path[0])
                if abs(PID_W(direction, yaw)) > 0.2:
                    state = TURN
                    HAL.setV(0)
            continue

        HAL.setV(PID_V(fwd_offset))
        HAL.setW(PID_W(direction, yaw))

    elif state == TURN:
        HAL.setV(0)
        HAL.setW(PID_W(direction, yaw))
        if abs(PID_W(direction, yaw)) < 0.2:
            HAL.setW(0)
            state = FORWARD

    GUI.showNumpy(grid)