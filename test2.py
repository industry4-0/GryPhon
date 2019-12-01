from ctypes import *
import math, argparse
import random
import os,sys
import cv2
import numpy as np
import time
import darknet
import sys, select
from objloader_simple import *

MIN_MATCHES = 100
resolution = 608
cases = 2

def render(img, obj, projection, model, x,y,w,h, color=False,scale=5,translation=0,x_tr=0):
    """
    Render a loaded obj model into the current video frame
    """
    print('RENDER')
    # colors = [(0,255,255),(0,0,255)]

    vertices = obj.vertices
    scale_matrix = np.eye(3)*scale
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0]+x_tr, p[1]-50+translation, p[2]] for p in points])
        # print(points[1])
        # print(points[0])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (0,255,255))
        else:
            cv2.fillConvexPoly(img, imgpts, (0,0,255))

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


def modelRender(frame,detection,bf,des_model,kp_model,orb,obj,model,camera_parameters,args,scale=5,translation=0,x_tr=0,color=False):

    x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
    homography = None

    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    if not np.shape(des_frame):
        pass
    elif np.shape(des_frame)[-1] == np.shape(des_model)[-1]:
        matches_1 = bf.match(des_model, des_frame)

        matches = []

        for m in matches_1:
            if m.distance > 0.80 * m.distance:
                matches.append(m)
        print(len(matches))
        #compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES:
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)
                # connect them with lines  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)  
                    # project cube or model
                    frame = render(frame, obj, projection, model,x,y,w,h, color,scale,translation,x_tr)
                    #frame = render(frame, model, projection)
                except:
                    print('ERroar')
                    pass
            
        else:
            
            print ("%d/%d" % (len(matches), MIN_MATCHES))
    else:
        print ("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    return frame



def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    bollflag = True
    scale = 5
    start = 0
    translation = 0
    x_tr = 0
    color = False
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


    global metaMain, netMain, altNames
    configPath = "./sev2/machine.cfg"
    weightPath = "./sev2/machine_last.weights"
    metaPath = "./sev2/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)
    cap.set(3, resolution)
    cap.set(4, resolution)

    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    orb = cv2.ORB_create() 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    model_ext = cv2.imread('/home/ikansizo/Desktop/augmented-reality-master/presentat.png')
    kp_model_ext, des_model_ext = orb.detectAndCompute(model_ext, None)
    model_in = cv2.imread('/home/ikansizo/Desktop/augmented-reality-master/presentat.png')
    kp_model_in, des_model_in = orb.detectAndCompute(model_in, None)
    obj = OBJ('/home/ikansizo/Desktop/augmented-reality-master/4sev/external/Outside_1.obj', swapyz=True)

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        
        if not bollflag:
            if time.time()-start <= 8:
                obj = OBJ('/home/ikansizo/Desktop/augmented-reality-master/4sev/external/Outside_1.obj', swapyz=True)
                model = model_ext
                kp_model = kp_model_ext
                des_model = des_model_ext
                color = False
            elif time.time()-start >8 and time.time()-start<=16:
                obj = OBJ('/home/ikansizo/Desktop/augmented-reality-master/4sev/external/Outside_3.obj', swapyz=True)
                color = False
            elif time.time()-start>16 and time.time()-start<=24:
                obj = OBJ('/home/ikansizo/Desktop/augmented-reality-master/screw_all_.obj', swapyz=True)
                color = False
            elif time.time()-start>24 and time.time()-start<=32:
                obj = OBJ('/home/ikansizo/Desktop/augmented-reality-master/Inside_1.obj', swapyz=True)
                scale = 8
                translation = 50
                color = True
            elif time.time()-start>32 and time.time()-start<=40:
                obj = OBJ('/home/ikansizo/Desktop/augmented-reality-master/Inside_3.obj', swapyz=True)
                scale = 8
                translation = 50
                color = True
                
        else:
            obj = OBJ('/home/ikansizo/Desktop/augmented-reality-master/4sev/external/Outside_1.obj', swapyz=True)
            model = model_ext
            kp_model = kp_model_ext
            des_model = des_model_ext

        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.40)
        if len(detections)>=1:
            if len(detections[0][2])>=4:
                if bollflag:
                    start = time.time()
                    bollflag = False
                frame_read = modelRender(frame_read,detections[0],bf,des_model,kp_model,orb,obj,model,camera_parameters,args,scale,translation,x_tr,color)
        image = cvDrawBoxes(detections, frame_read)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', frame_read)
        cv2.waitKey(3)

    cap.release()
    out.release()

if __name__ == "__main__":   
    YOLO()
