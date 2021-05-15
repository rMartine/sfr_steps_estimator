# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:05:09 2021

@author: rober
"""

"""
Loads and displays a video.
"""

# Importing OpenCV
import cv2
import cv2.aruco as aruco
import numpy as np
import math

# camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
camera_parameters = np.array([[1805.9811001658015, 0.0, 929.2455119852768], [0.0, 1849.9792459639896, 1180.2121331843236], [0.0, 0.0, 1.0]])
 


def find_squares(img, filteredImg, ratio, verticalArucoSide):
    # Applying thresholding and finding contours
    ret, thresh = cv2.threshold(filteredImg, 240 , 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    
    # Draw contours
    for contour in contours:
        # Aproximate squares
        epsilon = 0.1*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        
        rect = cv2.minAreaRect(approx)
        boxPrec = cv2.boxPoints(rect)
        box = np.int0(boxPrec)
        
        #Deciding if it is a box or a "line"
        # ratio = 20 # The higher the number the higher the difference between the larger side and the shortest side
        p1 = boxPrec[0]
        p2 = boxPrec[1]
        p3 = boxPrec[2]
        
        catAS1 = p2[0] - p1[0]
        catBS1 = p2[1] - p1[1]
        side1 = math.sqrt(math.pow(catAS1, 2) + math.pow(catBS1, 2))
        
        catAS2 = p2[0] - p3[0]
        catBS2 = p2[1] - p3[1]
        side2 = math.sqrt(math.pow(catAS2, 2) + math.pow(catBS2, 2))
        
        if (side1 > side2):
            if (side2 > 0):
                if ((side1/side2) >= ratio):
                    dst, pX, pY = get_distance_aruco_step(verticalArucoSide, box)
                    if (not(pX < 0 or pY < 0)):
                        cv2.drawContours(img,[box],0,(0,0,255),1)
                        img = cv2.circle(img, (int(pX), int(pY)), radius=4, color=(255, 0, 0), thickness=-1)
                        font = cv2.FONT_HERSHEY_COMPLEX
                        tag =  "h = %.2f in" % (dst)
                        cv2.putText(img, tag, (int(pX) + 7, int(pY)), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                dst, pX, pY = get_distance_aruco_step(verticalArucoSide, box)
                if (not(pX < 0 or pY < 0)):
                    cv2.drawContours(img,[box],0,(0,0,255),1)
                    img = cv2.circle(img, (int(pX), int(pY)), radius=4, color=(255, 0, 0), thickness=-1)
                    font = cv2.FONT_HERSHEY_COMPLEX
                    tag =  "h = %.2f in" % (dst)
                    cv2.putText(img, tag, (int(pX) + 7, int(pY)), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
        else:
            if (side1 > 0):
                if ((side2/side1) >= ratio):
                    dst, pX, pY = get_distance_aruco_step(verticalArucoSide, box)
                    if (not(pX < 0 or pY < 0)):
                        cv2.drawContours(img,[box],0,(0,0,255),1)
                        img = cv2.circle(img, (int(pX), int(pY)), radius=4, color=(255, 0, 0), thickness=-1)
                        font = cv2.FONT_HERSHEY_COMPLEX
                        tag =  "h = %.2f in" % (dst)
                        cv2.putText(img, tag, (int(pX) + 7, int(pY)), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            else:
                if (side1 == 0 and side2 > 0):
                    dst, pX, pY = get_distance_aruco_step(verticalArucoSide, box)
                    if (not(pX < 0 or pY < 0)):
                        cv2.drawContours(img,[box],0,(0,0,255),1)
                        img = cv2.circle(img, (int(pX), int(pY)), radius=4, color=(255, 0, 0), thickness=-1)
                        font = cv2.FONT_HERSHEY_COMPLEX
                        tag =  "h = %.2f in" % (dst)
                        cv2.putText(img, tag, (int(pX) + 7, int(pY)), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
    
    return img

def get_distance_aruco_step(verticalArucoSide, box):
    # print (box)
    # topSide, vertSide = get_box_top_side(box)
    arucoHeightIn = 3.75
    intPntX, intPntY = get_intersection(verticalArucoSide[0], verticalArucoSide[1], box[0], box[2])
    
    catArucA = verticalArucoSide[0][0] - verticalArucoSide[1][0]
    catArucB = verticalArucoSide[0][1] - verticalArucoSide[1][1]
    arucoHeight = math.sqrt(math.pow(catArucA, 2) + math.pow(catArucB, 2))
    
    arucoTopX = verticalArucoSide[0][0]
    arucoTopY = verticalArucoSide[0][1]
    if (verticalArucoSide[0][1] > verticalArucoSide[1][1]):
        arucoTopX = verticalArucoSide[1][0]
        arucoTopY = verticalArucoSide[1][1]
    catStepA = intPntX - arucoTopX
    catStepB = intPntY - arucoTopY
    stepHeight = math.sqrt(math.pow(catStepA, 2) + math.pow(catStepB, 2))
    
    arucoPixelInRatio = arucoHeightIn / arucoHeight
    stepHeightIn = (arucoHeight + stepHeight) * arucoPixelInRatio
    
    return stepHeightIn, intPntX, intPntY

def get_box_top_side (box):
    slopeS1 = get_slope(box[0], box[1])
    slopeS2 = get_slope(box[1], box[2])
    slopeS3 = get_slope(box[2], box[3])
    slopeS4 = get_slope(box[3], box[0])
    
    slopeS1 = abs_slope(slopeS1)
    slopeS2 = abs_slope(slopeS2)
    slopeS3 = abs_slope(slopeS3)
    slopeS4 = abs_slope(slopeS4)
    
    dtype = [('slope', float), ('pnt1', int), ('pnt2', int)]
    values = [(slopeS1, 0, 1),
              (slopeS2, 1, 2),
              (slopeS3, 2, 3),
              (slopeS4, 3, 4)]
    slopes = np.array(values, dtype = dtype)
    sortedSlopes = np.sort(slopes, axis = 0, order = ['slope'])
    tallestYPnt1 = box[sortedSlopes[0][1]][1] if (box[sortedSlopes[0][1]][1] < box[sortedSlopes[0][2]][1]) else box[sortedSlopes[0][2]][1]
    tallestYPnt2 = box[sortedSlopes[1][1]][1] if (box[sortedSlopes[1][1]][1] < box[sortedSlopes[1][2]][1]) else box[sortedSlopes[1][2]][1]
    
    topSide = None
    verticalSide = [box[sortedSlopes[2][1]], box[sortedSlopes[2][2]]]
    
    if (tallestYPnt1 < tallestYPnt2):
        topSide = [box[sortedSlopes[0][1]], box[sortedSlopes[0][2]]]
    else:
        topSide = [box[sortedSlopes[1][1]], box[sortedSlopes[1][2]]]
    
    return topSide, verticalSide

def abs_slope(slope):
    return (math.sqrt(math.pow(slope, 2)))

def get_slope(point1, point2):
    slope = np.nan
    if ((point2[0] - point1[0]) != 0):
        slope = (point2[1] - point1[1])/(point2[0] - point1[0])
    return slope

def get_intersection(pntv1, pntv2, pnth1, pnth2):
    mv = get_slope(pntv1, pntv2)
    mh = get_slope(pnth1, pnth2)
    X = (((mv * pntv1[0]) - pntv1[1] - (mh * pnth1[0]) + pnth1[1])/(mv - mh))
    Y = ((mv * X) - (mv * pntv1[0]) + pntv1[1])
    
    return X, Y

def find_lines(img, filteredImg):
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(filteredImg, low_threshold, high_threshold)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((x2-x1) > (y2-y1)):
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)
            
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    return lines_edges

def findArucoBox(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
    
    arucoPoints = None
    if np.all(ids != None):
        # frame = aruco.drawDetectedMarkers(frame, corners)
        x1 = (corners[0][0][0][0], corners[0][0][0][1])
        x2 = (corners[0][0][1][0], corners[0][0][1][1])
        x3 = (corners[0][0][2][0], corners[0][0][2][1])
        x4 = (corners[0][0][3][0], corners[0][0][3][1])
        arucoPoints = np.array([x1, x2, x3, x4])
        
    return arucoPoints
    

if __name__ == "__main__":
    # Get image from photo
    image = cv2.imread('photos/test/20210111_200358.jpg')
    image = cv2.resize(image, (int(image.shape[1] * 0.4), int(image.shape[0] * 0.4)), interpolation = cv2.INTER_AREA)
    
    # get ARUCO box
    arucoBox = findArucoBox(np.copy(image))
    topSideArucoBox, verticalSideArucoBox = get_box_top_side(arucoBox)
    
    # Converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to find squares
    diameter = 9 # Diameter of each pixel neighborhood
    sigmaColor = 30 # Filter sigma in the color space
    sigmaSpace = 50 # Filter sigma in the coordinate space
    bilateral_gray = cv2.bilateralFilter(gray, diameter, sigmaColor, sigmaSpace)
    
    # Gausian filter to find lines
    kernel_size = 9
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    
    # Canny filters to get contours from both images
    low_threshold = 50
    high_threshold = 150
    edgesBlur = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # edgesBilateral = cv2.Canny(bilateral_gray, low_threshold, high_threshold)
    
    # Get images and polygons
    imgSquares = find_squares(np.copy(image), edgesBlur, 60, verticalSideArucoBox)
    # imgLines = find_lines(np.copy(image), edgesBlur)
        
    
    # Stacking the images to print them together for comparison
    images = np.hstack((image, imgSquares))
    
    # Display the resulting frame
    cv2.imshow('Frame', imgSquares)
    cv2.waitKey(0)
    
    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
        # break
    # Break the loop
    # else:
        # break
    
    image_name = 'stair_result/processed.png'
    cv2.imwrite(image_name, imgSquares)
    
    # Closes all the frames
    cv2.destroyAllWindows()