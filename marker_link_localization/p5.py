import GUI
import HAL
import yaml
from pathlib import Path
import pyapriltags
import cv2
import numpy as np
import copy
import random as rand

tag_width = 0.24
tag_height = 0.24

try:
    conf = yaml.safe_load(
        Path("/resources/exercises/marker_visual_loc/apriltags_poses.yaml").read_text()
    )
    tags = conf["tags"]
except FileNotFoundError:
    print("[ERROR] El archivo YAML no se encontró. Verifique la ruta.")
    conf = {"tags": {}}
    tags = conf["tags"]
except yaml.YAMLError as e:
    print(f"[ERROR] Error al procesar el archivo YAML: {e}")
    conf = {"tags": {}}
    tags = conf["tags"]

image = HAL.getImage()

size = image.shape
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)

matrix_camera = np.array(
[[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
dtype="double",
)

dist_coeffs = np.zeros((4, 1))

def detect_tag(image):
    tag_corners_2d = []
    tags_positions_3d = []

    detector = pyapriltags.Detector(searchpath=["apriltags"], families="tag36h11")

    print("[INFO] loading image...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("[INFO] detecting AprilTags...")
    results = detector.detect(gray)
    print("[INFO] {} total AprilTags detected".format(len(results)))

    # loop over the AprilTag detection results
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        tag_corners_2d.append(r.corners)
        tags_3d = str(r.tag_id)
        tags_positions_3d.append(tags["tag_"+tags_3d]['position'])

        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(
            image,
            tagFamily,
            (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        print("[INFO] tag family: {}".format(tagFamily))

    GUI.showImage(image)
    return tag_corners_2d, tags_positions_3d        

def get_tag_corners():
    #The apriltags are set at 0.8 meters in height and have a size of 0.3 x 0.3 meters.
    x, y, z = tag_width, tag_height, 0
    corners = np.array(
        [
            [-x / 2, y / 2, z],
            [x / 2, y / 2, z],
            [x / 2, -y / 2, z],
            [-x / 2, -y / 2, z],
        ]
    )

    return corners

def get_tf_beacon2camera(img_tag_corner, tag_corners):
    # Usar el método SOLVEPNP_IPPE_SQUARE
    _, rvec, tvec = cv2.solvePnP(
        tag_corners, img_tag_corner, matrix_camera, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    # Obtener la matriz de rotación
    R, _ = cv2.Rodrigues(rvec)

    matrix_camera2beacon = np.eye(4)  # Matriz identidad 4x4
    matrix_camera2beacon[:3, :3] = R  # Insertar la matriz de rotación
    matrix_camera2beacon[:3, 3] = tvec.flatten()  # Insertar el vector de traslación

    # Rotación -90° en X
    R_X = np.array([
        [ 1,  0,  0, 0],
        [ 0, 0,  1, 0],
        [ 0, -1, 0, 0],
        [ 0,  0,  0, 1]
    ])
    # Rotación -90° en Z
    R_Z = np.array([
        [ 0,  1,  0, 0],
        [-1,  0,  0, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1]
    ])

    R = np.dot(R_Z,R_X)

    # Aplicar las rotaciones a la matriz de transformación

    matrix_beacon2camera = np.linalg.inv(matrix_camera2beacon)

    matrix_beacon2camera = np.dot(R, matrix_beacon2camera)

    return matrix_beacon2camera

def get_tf_world2beacon(tag_pose3d, tag_angle):
    # Obtener la posición y orientación de la baliza
    x, y, z = tag_pose3d

    # Crear la matriz de transformación (4x4)
    matrix_world2beacon = np.eye(4)  # Matriz identidad 4x4

    # Asignar la posición (traslación)
    matrix_world2beacon[0, 3] = x
    matrix_world2beacon[1, 3] = y
    matrix_world2beacon[2, 3] = z

    # Crear la matriz de rotación en torno a Z
    R_z = np.array([
        [np.cos(tag_angle), -np.sin(tag_angle), 0],
        [np.sin(tag_angle),  np.cos(tag_angle), 0],
        [0,              0,             1]
    ])
    matrix_world2beacon[:3, :3] = R_z

    return matrix_world2beacon

def navigation(position,yaw_pos):
    
    HAL.setW(rand.uniform(-1, 1))
    HAL.setV(rand.uniform(0, 0.5))

    GUI.showEstimatedPose((position[0], position[1], yaw_pos+np.pi/2))

    print("Position:", position)

while True:
    # Enter iterative code!
    img_tag_corners = []
    tags_pose3d = []
    img = HAL.getImage()
    img_tag_corners, tags_pose3d = detect_tag(img)

    tags_pose3d = copy.deepcopy(tags_pose3d)

    if len(tags_pose3d) == 0 or len(img_tag_corners) == 0:
        HAL.setW(0.3)
        HAL.setV(0)

    for img_tag_corner,tag_pose3d in zip(img_tag_corners, tags_pose3d):
        tag_corners = []
        tag_corners = get_tag_corners()
        tag_angle = copy.deepcopy(tag_pose3d[2])
        tag_pose3d[2] = 0.8

        matrix_beacon2camera = get_tf_beacon2camera(img_tag_corner, tag_corners)
        matrix_world2beacon = get_tf_world2beacon(tag_pose3d, tag_angle)

        matrix_world2camera = np.dot(matrix_world2beacon, matrix_beacon2camera)

        pitch_pos = np.arctan2(-matrix_world2camera[2, 0], np.sqrt(matrix_world2camera[0, 0] * matrix_world2camera[0, 0] + matrix_world2camera[1, 0] * matrix_world2camera[1, 0]))

        yaw_pos = np.arctan2(matrix_world2camera[1, 0] / np.cos(pitch_pos), matrix_world2camera[0, 0] / np.cos(pitch_pos))

        position = matrix_world2camera[:3, 3]

        navigation(position, yaw_pos)
