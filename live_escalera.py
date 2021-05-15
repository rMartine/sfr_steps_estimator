import argparse
import numpy as np
# from OpenGL.GL import *
# from glyphs import Glyphs
import cv2
import cv2.aruco as aruco
import os
import math
from objloader_simple import *

def aruco_model():
    homography = None
    # matrix of camera parameters (made up but works quite well for me)
    # camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    camera_parameters = np.array([[1805.9811001658015, 0.0, 929.2455119852768], [0.0, 1849.9792459639896, 1180.2121331843236], [0.0, 0.0, 1.0]])
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    # Load 3D model from OBJ file
    obj = OBJ('models/escalera_m.obj', swapyz=False)
    # obj = OBJ('models/column.obj', swapyz=False)
    print (obj)

    while True:
        #Read image from folder
        ret = False
        frame = None

        # frame = cv2.imread('photos/test/20210111_200612.jpg')
        frame = cv2.imread('photos/test/20210111_200358.jpg')
        # frame = cv2.imread('photos/calibration.jpg')
        # frame = cv2.imread(os.path.join(dir_name, 'photos/ott_hill.jpg'), 0)
        if frame is not None:
            ret = True

        if not ret:
            print ('No photo to process')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)

        if np.all(ids != None):
            # frame = aruco.drawDetectedMarkers(frame, corners)
            x1 = (corners[0][0][0][0], corners[0][0][0][1])
            x2 = (corners[0][0][1][0], corners[0][0][1][1])
            x3 = (corners[0][0][2][0], corners[0][0][2][1])
            x4 = (corners[0][0][3][0], corners[0][0][3][1])

            aruco_original = cv2.imread('marker/model.jpg')
            size = aruco_original.shape
            pts_aruco_warped = np.array([x1, x2, x3, x4])
            pts_aruco_original = np.array(
                [
                    [0, 0],
                    [size[1] - 1, 0],
                    [size[1] - 1, size[0] - 1],
                    [0, size[0] - 1]
                ], dtype=float
            )
            homography, status = cv2.findHomography(pts_aruco_original, pts_aruco_warped, cv2.RANSAC, 5.0)
            if homography is not None:
                try:
                    projection = projection_matrix(camera_parameters, homography)
                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = render(frame, obj, projection, cv2.cvtColor(aruco_original, cv2.COLOR_BGR2GRAY), False)
                except:
                    pass
        height = int(frame.shape[0]/1.6)
        width = int(frame.shape[1]/1.6)
        frame = cv2.resize(frame, (height, width))
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    image_name = 'stair_result/processed.png'
    cv2.imwrite(image_name, frame)
    cv2.destroyAllWindows()
    return 0

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 300 * 2.905978494623656
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        # points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        points = np.array([[p[0] + w / 2, p[1] + h, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    # project_image()
    aruco_model()

def project_image():
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
        arucoParameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
        if np.all(ids != None):
            display = aruco.drawDetectedMarkers(frame, corners)
            x1 = (corners[0][0][0][0], corners[0][0][0][1])
            x2 = (corners[0][0][1][0], corners[0][0][1][1])
            x3 = (corners[0][0][2][0], corners[0][0][2][1])
            x4 = (corners[0][0][3][0], corners[0][0][3][1])

            im_dst = frame
            im_src = cv2.imread("photos/ott_hill.jpg")
            size = im_src.shape
            pts_dst = np.array([x1, x2, x3, x4])
            pts_src = np.array(
                [
                    [0, 0],
                    [size[1] - 1, 0],
                    [size[1] - 1, size[0] - 1],
                    [0, size[0] - 1]
                ], dtype=float
            )

            h, status = cv2.findHomography(pts_src, pts_dst)
            temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
            cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
            im_dst = im_dst + temp
            cv2.imshow('Display', im_dst)
        else:
            display = frame
            cv2.imshow('Display', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
