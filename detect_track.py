# Kevin Liu (6/9/2022)

import os
import cv2
import numpy as np
import matplotlib.pyplot as pl
import math

CANNY_THRESHOLD_LO = 50
CANNY_THRESHOLD_HI = 150

MIN_TRACK_LEN = 120

# area of interest
LEFT_BOUNDARY = 200
RIGHT_BOUNDARY = 1100
MID = 600
TOP = 200

def generate_coordinates(image, line_parameters):

    # show line from bottom to top on the given percentage of the screen
    SHOW_PERCENTAGE = 4.5/10

    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * SHOW_PERCENTAGE)

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), deg=1)                    # fit a polynomial of degree one

        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    left_line = generate_coordinates(image, left_fit_avg)
    right_line = generate_coordinates(image, right_fit_avg)

    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)                           # blurs image to reduce noice
    canny = cv2.Canny(blur, CANNY_THRESHOLD_LO, CANNY_THRESHOLD_HI)    # Trace out gradient difference
    return canny

def region_of_interest(image_param):
    # area of interest (check matplotlib for value)
    HEIGHT = image_param.shape[0] - 1 # size when lim(y->0)

    # create white triangle on black mask
    triangle = np.array([
        [(LEFT_BOUNDARY, HEIGHT), (RIGHT_BOUNDARY, HEIGHT), (MID, TOP)]
        ])
    mask = np.zeros_like(image_param)
    cv2.fillPoly(mask, triangle, 255)

    # bitwise overlays the entire image with binary 0 except for the area of interest (binary 1)
    masked_image = cv2.bitwise_and(image_param, mask)
    return masked_image


def display_track(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return line_image

def calc_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

image = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_image.jpg'))
lane_image = np.copy(image)



# IMAGE PROCESSING =============================================================================================================
# canny_img = canny(lane_image)
# cropped_iamge = region_of_interest(canny_img)

# # cv2.HoughLinesP(image, precision_pixel_count, precision_in_radians, threshold_for_bin, empty_np_array)
# lines = cv2.HoughLinesP(cropped_iamge, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# avg_lines = average_slope_intercept(lane_image, lines)
# line_image = display_track(lane_image, avg_lines)

# # combine lanes drawn and the original image
# resulting_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

# cv2.imshow('Test Image', resulting_image)
# cv2.imshow('Computer Vision Gradient', canny_img)

# # pl.imshow(combo_image)
# # pl.show()

# cv2.waitKey(0)


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    canny_img = canny(frame)
    cropped_iamge = region_of_interest(canny_img)

    # cv2.HoughLinesP(image, precision_pixel_count, precision_in_radians, threshold_for_bin, empty_np_array)
    lines = cv2.HoughLinesP(cropped_iamge, 2, np.pi/180, 100, np.array([]), minLineLength=MIN_TRACK_LEN, maxLineGap=5)
    # avg_lines = average_slope_intercept(frame, lines)
    line_image = display_track(frame, lines)

    # combine lanes drawn and the original image
    resulting_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('Test Image', resulting_image)
    cv2.imshow('Computer Vision Gradient', canny_img)

    # pl.imshow(combo_image)
    # pl.show()

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()