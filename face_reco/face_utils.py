import math
import numpy as np
import cv2


def get_distance(p1, p2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))


def get_angle_z(shape):
    leftEyePts = shape[36:42]
    rightEyePts = shape[42:48]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    angle = abs(angle)
    z_angle = abs(angle - 180)
    # print("z angle", z_angle)
    return z_angle


def get_angle_y(shape):
    eye_center = ((shape[36][0] + shape[45][0]) // 2, (shape[36][1] + shape[45][1]) // 2)
    mouth_center = ((shape[48][0] + shape[54][0]) // 2, (shape[48][1] + shape[54][1]) // 2)
    dist1 = get_distance(eye_center, shape[30])
    dist2 = get_distance(shape[30], mouth_center)
    y_angle = dist1 / dist2
    y_angle = abs(y_angle)
    if y_angle > 1:
        y_angle = 1 / y_angle
        y_angle = 1 - y_angle
        y_angle = -1 * math.asin(y_angle) * (180 / math.pi)
        y_angle *= get_division(y_angle)
    else:
        y_angle = 1 - y_angle
        y_angle = math.asin(y_angle) * (180 / math.pi)
        y_angle *= get_division(y_angle)
    return y_angle


def get_division(y_angle):
    y_angle = abs(y_angle)
    # print(y_angle)
    if y_angle < 20:
        return 0.1
    if y_angle < 40:
        return 0.3
    elif y_angle > 50:
        return 0.6
    else:
        return 0.8


def get_angle_x(shape):
    dist1 = get_distance(shape[0], shape[27])
    dist2 = get_distance(shape[27], shape[16])
    x_angle = dist1 / dist2
    x_angle = abs(x_angle)
    if x_angle > 1:
        x_angle = 1 / x_angle
        x_angle = 1 - x_angle
        x_angle = -1 * math.asin(x_angle) * (180 / math.pi)
        x_angle *= get_multiplier(x_angle)
        if x_angle < -60:
            x_angle = -60
    else:
        x_angle = 1 - x_angle
        x_angle = math.asin(x_angle) * (180 / math.pi)
        x_angle *= get_multiplier(x_angle)
        if x_angle > 60:
            x_angle = 60
    # print("xangle", abs(x_angle))
    return x_angle


def get_multiplier(x_angle):
    x_angle = abs(x_angle)
    if x_angle < 20:
        return 0.5
    if x_angle < 40:
        return 0.7
    elif x_angle > 50:
        return 0.85
    else:
        return 0.8


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
