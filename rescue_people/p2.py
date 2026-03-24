import GUI
import HAL
import time
import cv2
import utm

# Boat geographic coordinates in decimal degrees
lat_boat = 40.28006
lon_boat = -3.81764

# Convert geographic coordinates to UTM
utm_boat = utm.from_latlon(lat_boat, lon_boat)
utm_origin_x = utm_boat[0]  # Boat easting
utm_origin_y = utm_boat[1]  # Boat northing
ZONE_NUMBER = utm_boat[2]   # UTM zone number of the boat
ZONE_LETTER = utm_boat[3]   # UTM zone letter of the boat

VICTIMS_X = 30
VICTIMS_Y = -40
BOAT_X = 0
BOAT_Y = 0
Z = 1.3
AZ = 0

x_vel = 0.25
angle = 0.6
iterations = 0
spiral_iterations = 300
error_margin = 0.2

start_time = 0

HAL.takeoff(3)

current_altitude = HAL.get_position()[2]  # Get the current altitude (z-axis)
if abs(current_altitude - 3) < error_margin:
    start_time = time.time()  # Record the time when the altitude of 1.3 meters is reached
            
HAL.set_cmd_pos(VICTIMS_X, VICTIMS_Y, 2, 0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def xy_to_gps(x, y):
    # Convert UTM coordinates to GPS coordinates
    lat, lon = utm.to_latlon(x, y, ZONE_NUMBER, ZONE_LETTER)
    return lat, lon

def gazebo_to_utm(x_gazebo, y_gazebo, utm_origin_x, utm_origin_y):
    # Add Gazebo relative coordinates to the UTM origin
    utm_x = utm_origin_x + x_gazebo
    utm_y = utm_origin_y + y_gazebo
    return utm_x, utm_y

def detect_persons(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1,
    minNeighbors=5,
    minSize=(15, 15),
    maxSize=(50, 50))
    
    for (x, y, w, h) in faces:
        
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]

        return True
        
    return False

def spiral_path():

    x = VICTIMS_X
    y = VICTIMS_Y

    side_length = 1  # Initial side length of the square in meters
    increment = 1.1  # How much the side increases for each iteration of the spiral
    turns = 8  # Number of turns in the spiral

    spiral_points = []

    # Loop to generate the shrinking square spiral
    for turn in range(turns):
        
        side_length += increment

        x += side_length
        spiral_points.append((x, y))

        y += side_length
        spiral_points.append((x, y))
        
        side_length += increment

        x -= side_length
        spiral_points.append((x, y))

        y -= side_length
        spiral_points.append((x, y))

        side_length += increment

    return spiral_points
    
def rotate_image(image):
    # Get the center of the image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, 20, 1.0)

    # Apply the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated_image

def calculate_distance(pos1, pos2):
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def spiral():
    global start_time

    point_index = 0
    face_counter = 0
    persons = []

    points = spiral_path()
    
    while point_index < len(points):
        # Check if 8 minutes (480 seconds) have passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= 480:
          print("Battery low, returning to base")
          return False
          
        ventral_image = HAL.get_ventral_image()

        rotated_image = ventral_image
        for i in range(17):
          rotated_image = rotate_image(rotated_image)
        
          if detect_persons(rotated_image):
            current_position = HAL.get_position()
            should_add = True

            for pos in persons:
                if calculate_distance(current_position, pos) < 3:
                    should_add = False
                    break

            if should_add:
                # Convert Gazebo coordinates to UTM
                utm_x, utm_y = gazebo_to_utm(current_position[0], current_position[1], utm_origin_x, utm_origin_y)
                # Convert UTM coordinates to GPS
                lat, lon = xy_to_gps(utm_x, utm_y)
                
                # Determine N/S for latitude and E/W for longitude
                lat_direction = 'N' if lat >= 0 else 'S'
                lon_direction = 'E' if lon >= 0 else 'W'
                              
                GUI.showImage(rotated_image)
                
                persons.append(current_position)
                face_counter += 1
                
                print(f"Person {face_counter} detected at: Latitude: {abs(lat):.6f}° {lat_direction}, Longitude: {abs(lon):.6f}° {lon_direction}")

        GUI.showLeftImage(HAL.get_ventral_image())
        position = points[point_index]
        HAL.set_cmd_pos(position[0], position[1], Z, AZ)
        if abs(HAL.get_position()[0] - position[0]) < error_margin and abs(HAL.get_position()[1] - position[1]) < error_margin:
            point_index += 1
            
    return True

while True:
    GUI.showImage(HAL.get_frontal_image())
    GUI.showLeftImage(HAL.get_ventral_image())
    # When it reaches the victims' position, start the spiral
    if abs(HAL.get_position()[0] - VICTIMS_X) < error_margin and abs(HAL.get_position()[1] - VICTIMS_Y) < error_margin:
      if spiral() == False:
        while abs(HAL.get_position()[0] - BOAT_X) > 0.05 and abs(HAL.get_position()[1] - BOAT_Y) > 0.05:
            HAL.set_cmd_pos(BOAT_X, BOAT_Y, Z, AZ) # Command to return to the boat

        HAL.land()
