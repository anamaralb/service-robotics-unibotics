import GUI
import HAL
import math
import time
import numpy as np

SEARCHING = 0
SPOT_FOUND = 1
PARKING = 2

def aproach_distance(right_lasers,front_lasers,back_lasers):
  min_distance = math.inf

  lasers=[right_lasers,front_lasers,back_lasers]
  
  lasers = np.concatenate(lasers, axis=0)
  
  valid_distances = []
  
  for dist in lasers:
        if not np.isnan(dist) and not np.isinf(dist):
            valid_distances.append(dist)
            
  for laser in valid_distances:
    if laser < min_distance:
      min_distance = laser
      
  return min_distance

def p_aproach_controller(current_distance):
    target_distance = 0.5
        
    error = target_distance - current_distance
    
    Kp_angular = 0.1

    angular_velocity = Kp_angular * error    
    
    HAL.setV(1)
    HAL.setW(angular_velocity)

def p_orientation_controller(target, current, Kp=2):
    error = (target - current + math.pi) % (2 * math.pi) - math.pi
    return Kp * error

def get_street_direction():
    directions = []
    confidences = []
    labels = []
    car_orientation = HAL.getPose3d().yaw

    for laser_data, label, offset in [
        (HAL.getRightLaserData().values, "right", -math.pi/2),
        (HAL.getFrontLaserData().values, "front", 0),
        (HAL.getBackLaserData().values, "back", -math.pi),
    ]:
        _, xy = parse_laser_data(laser_data)
        if len(xy) > 0:
            center = np.mean(xy, axis=0)
            points_centered = xy - center
            _, s, Vt = np.linalg.svd(points_centered)
            directions.append(Vt[0])
            confidences.append(s[0])
            labels.append((offset, label))

    best_index = np.argmax(confidences)
    direction = directions[best_index]
    offset, _ = labels[best_index]
    theta = np.arctan2(direction[1], direction[0])
    return (theta + car_orientation + offset + math.pi) % (2 * math.pi) - math.pi

def parse_laser_data(laser_data):
    polar = []
    cartesian = []
    for i in range(len(laser_data)):
        dist = laser_data[i]
        if np.isnan(dist) or np.isinf(dist):
            continue
        angle = math.radians(i - 90)
        polar.append((dist, angle))
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        cartesian.append([x, y])
    return polar, np.array(cartesian)

def spot_obstacles():
    x_min = 4.5
    x_max = 7
    y_min = -2.75
    y_max = 2.75
    
    laser_lat = HAL.getRightLaserData().values
    _, parsed_laser = parse_laser_data(laser_lat)

    for point in parsed_laser:
        if x_min < point[0] and point[0] < x_max: 
            return True
        if y_min < point[1] and point[1] < y_max:
            return True
    return False

def reverse_turn():
    
    orientacion_inicial = HAL.getPose3d().yaw
    orientacion_objetivo = orientacion_inicial + math.radians(40)

    if orientacion_objetivo > math.pi:
        orientacion_objetivo -= 2 * math.pi
    elif orientacion_objetivo < -math.pi:
        orientacion_objetivo += 2 * math.pi

    velocidad_angular = 0.7   
    velocidad_lineal = -0.5   

    while True:
        orientacion_inicial = HAL.getPose3d().yaw
        error = orientacion_objetivo - orientacion_inicial

        if error > math.pi:
            error -= 2 * math.pi
        elif error < -math.pi:
            error += 2 * math.pi

        if abs(error) < math.radians(2):  
            HAL.setW(0)
            HAL.setV(0)
            break

        HAL.setW(velocidad_angular)
        HAL.setV(velocidad_lineal)
        time.sleep(0.1)  

def reverse_correction(back_lasers, distancia_seguridad=0.7, tiempo_maximo_retroceso=5.0):

    orientacion_objetivo = 0  
    velocidad_angular = -0.8  
    velocidad_lineal = -0.5  

    tiempo_inicio = time.time()

    while True:
        orientacion_actual = HAL.getPose3d().yaw
        error = orientacion_objetivo - orientacion_actual

        if error > math.pi:
            error -= 2 * math.pi
        elif error < -math.pi:
            error += 2 * math.pi

        mediciones_laser = HAL.getBackLaserData().values
        if mediciones_laser:
            distancia_minima = min(mediciones_laser)
            if distancia_minima < distancia_seguridad:
                HAL.setW(0)
                HAL.setV(0)
                deteccion_laser = True
                break
        else:
            if time.time() - tiempo_inicio > tiempo_maximo_retroceso:
                HAL.setW(0)
                HAL.setV(0)
                break

        HAL.setW(velocidad_angular * (error / abs(error) if error != 0 else 0))
        HAL.setV(velocidad_lineal)
        time.sleep(0.05)


def forward_correction(distancia_seguridad=2, tiempo_maximo_avance=7.0):
    tiempo_inicio = time.time()

    while True:
        orientacion_actual = HAL.getPose3d().yaw

        if orientacion_actual < 0:
            HAL.setW(-0.4)
        if orientacion_actual > 0:
            HAL.setW(0.4)

        HAL.setV(0.3)

        front_lasers = HAL.getFrontLaserData()
        valores_finitos = [v for v in front_lasers.values if not math.isinf(v)]

        if valores_finitos:
            distancia_minima = min(valores_finitos)
            if distancia_minima < distancia_seguridad:
                HAL.setW(0)
                HAL.setV(0)
                break
        else:
            if time.time() - tiempo_inicio > tiempo_maximo_avance:
                HAL.setW(0)
                HAL.setV(0)
                break

state = SEARCHING
search_started = False
spot_timer = 0
direction = HAL.getPose3d().yaw

while True:
    yaw = HAL.getPose3d().yaw
    right_lasers = HAL.getRightLaserData()
    front_lasers = HAL.getFrontLaserData()
    back_lasers = HAL.getBackLaserData()

    target_distance = aproach_distance(right_lasers.values,front_lasers.values,back_lasers.values)

    if state == SEARCHING:
        if target_distance > 1.3 and not search_started:
            p_aproach_controller(target_distance)
        else:
            HAL.setW(p_orientation_controller(get_street_direction(), yaw))
            search_started = True

        if (not spot_obstacles() and search_started):
            spot_timer = time.time()
            state = SPOT_FOUND
            print(">> Hueco detectado.")

    elif state == SPOT_FOUND:
        if time.time() - spot_timer > 10:
            HAL.setV(0)
            state = PARKING
            print(">> Avanzando hacia el hueco.")

    elif state == PARKING:
        reverse_turn()
        reverse_correction(back_lasers)
        forward_correction()
        exit()
