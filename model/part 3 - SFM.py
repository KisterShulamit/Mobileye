# ----------------------
# Finding the distance
# ----------------------

# ----------------
# manage_points.py
# ---------------

import math


def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def is_the_same_point(p1, p2):
    if dist(p1, p2) < 60:
        return True
    return False


def remove_duplicates(points):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if is_the_same_point(points[i], points[j]):
                points.remove(points[i])
                break


def get_prev_cur(absolute, cur):
    prev = []
    new_cur = []
    # for i in range(4):
    #     absolute[i]=absolute[i+1]
    # absolute[4] = cur.copy()
    save = cur.copy()
    # for i in range (3,2,-1):
    for p2 in cur:
        for p1 in absolute[0]:
            if is_the_same_point(p1, p2):
                prev.append(p1)
                new_cur.append(p2)
                absolute[0].remove(p1)
                # cur.remove(p2)
                break

    absolute[0] = save
    return prev, new_cur


# -------
#  SFM.py
# -------

import numpy as np
from numpy.core.fromnumeric import shape
import math
from scipy.fft import fft, ifft


# import SFM_standAlone

def calc_TFL_dist(prev_container, curr_container, focal, pp):
    print("calc_TFL_dist")
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize_pt(pt, focal, pp):
    return [(pt[0] - pp[0]) / focal, (pt[1] - pp[1]) / focal, 1]


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return np.array([normalize_pt(pt, focal, pp) for pt in pts])


def unnormalize_pt(pt, focal, pp):
    return [(pt[0] * focal) + pp[0], (pt[1] * focal) + pp[1], 1]


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return np.array([unnormalize_pt(pt, focal, pp) for pt in pts])


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    T = EM[:3, -1]
    foe = (T[0] / T[2], T[1] / T[2])
    return R, foe, T[2]


def rotate_pt(pt, R):
    abc = R.dot(pt)
    return [abc[0] / abc[2], abc[1] / abc[2], 1]


def rotate(pts, R):
    # rotate the points - pts using R
    return np.array([rotate_pt(pt, R) for pt in pts])


def distance_pt_from_line(pt, m, n):
    return abs((m * pt[0] - pt[1] + n) / math.sqrt(math.pow(m, 2) + 1))


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - p[0] * foe[1]) / (foe[0] - p[0])
    closest_pt = (0, norm_pts_rot[0])

    for index, pt in enumerate(norm_pts_rot[1:]):
        if distance_pt_from_line(pt, m, n) < distance_pt_from_line(closest_pt[1], m, n):
            closest_pt = (index + 1, pt)
    return closest_pt


def calc_dist(p_curr, p_rot, foe, tZ):
    Zx = tZ * ((foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0]))
    Zy = tZ * ((foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1]))

    x_diff = abs(p_rot[0] - p_curr[0])
    y_diff = abs(p_rot[1] - p_curr[1])

    return (x_diff / (x_diff + y_diff)) * Zx + (y_diff / (x_diff + y_diff)) * Zy


# -----------------------
# SFM_stanAlone.py
# -----------------------

import numpy as np
# import matplotlib._png as png
import matplotlib.pyplot as plt


# from Logical_unit.Finding_the_distance import SFM


class FrameContainer(object):
    def __init__(self):
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


# --------------------
# TFL_Manager.p
# --------------------

class TFL_Manager:
    def __init__(self):
        pass

    def __Identify_light_sources(self, image):
        # open part 1and get x and y
        red_x, red_y, green_x, green_y = find_tfl_lights_in_image(image)
        return red_x, red_y, green_x, green_y

    def __Selection_of_traffic_lights(self, image, red_x, red_y, green_x, green_y):
        red_x, red_y, green_x, green_y = get_traffic_light_points(image, red_x.copy(), red_y.copy(), green_x.copy(),
                                                                  green_y.copy())
        return red_x, red_y, green_x, green_y

    def __Finding_the_distance(self, im, red_x, red_y, green_x, green_y, pkl, absolute):
        x = np.concatenate([red_x, green_x])
        y = np.concatenate([red_y, green_y])
        points = list(zip(x, y))
        remove_duplicates(points)
        prev, curr = get_prev_cur(absolute, points)
        prev = np.array(prev)
        curr = np.array(curr)

        prev_container = FrameContainer()
        curr_container = FrameContainer()
        prev_container.traffic_light = prev
        curr_container.traffic_light = curr
        curr_container.EM = pkl['EM']
        curr_container = calc_TFL_dist(prev_container, curr_container, pkl['focal'], pkl['pp'])
        return prev_container, curr_container

    def parse_frame(self, image, pkl, absolute):
        red_x_1, red_y_1, green_x_1, green_y_1 = self.__Identify_light_sources(image)
        red_x_2, red_y_2, green_x_2, green_y_2 = self.__Selection_of_traffic_lights(image, red_x_1, red_y_1, green_x_1,
                                                                                    green_y_1)
        prev_container, curr_container = self.__Finding_the_distance(image, red_x_2, red_y_2, green_x_2, green_y_2, pkl,
                                                                     absolute)

        return {'part_1': {'red_x': red_x_1, 'red_y': red_y_1, 'green_x': green_x_1, 'green_y': green_y_1},
                'part_2': {'red_x': red_x_2, 'red_y': red_y_2, 'green_x': green_x_2, 'green_y': green_y_2},
                'part_3': {'prev_container': prev_container, 'curr_container': curr_container, 'focal': pkl['focal'],
                           'pp': pkl['pp']}}

