import cv2
import numpy as np
import dlib
from math import hypot, atan2, degrees, tan
import pandas as pd
import time
from scipy.ndimage.interpolation import shift
import re


def get_window(size, list_of_items, new_item):
    if len(list_of_items) < size:
        list_of_items.append(new_item)
    else:
        list_of_items.pop(0)
        list_of_items.append(new_item)
    return list_of_items

def inverse_weighted_ma(list_of_pixels):
    weights = np.arange(1,len(list_of_pixels) + 1)[::-1]
    return(np.dot(list_of_pixels, weights)/np.sum(weights))


def midpoint(p1 ,p2):
    return int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)

# calculates the ratio of (upper  to lower eyelids) and (left to right points) of an eye.
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    print("left point", left_point)
    left_upper = (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y)
    right_upper = (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y)


    left_lower = (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y)
    right_lower = (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)

    #print(midpoint(left_upper, right_upper))


    # # calculating on y parameter (not hypot)
    # denominator = left_point[0] - right_point[0]
    # numerator_1 = left_upper[1] - left_lower[1]
    # numerator_2 = right_upper[1] - right_lower[1]

    denominator = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))

    numerator_1 = hypot((left_upper[0] - left_lower[0]), (left_upper[1] - left_lower[1]))


    numerator_2 = hypot((right_upper[0] - right_lower[0]), (right_upper[1] - right_lower[1]))
    print("left upper",left_upper[1], "left lower", left_lower[1])
    print(numerator_1, "   ", numerator_2, "   ", denominator, "ratio components")

    ratio = (numerator_1 + numerator_2) / (denominator * 2)
    return ratio

def GetAngle(p1, p2, facial_landmarks):
    """
        Calculate the angle of the line going through 2 given points

        p1 and p2: landmark points
        facial_landmarks: predicted landmark locations
    """

    # get the x, y pixel locations for each point
    x1, y1 = (facial_landmarks.part(p1).x, facial_landmarks.part(p1).y)
    x2, y2 = (facial_landmarks.part(p2).x, facial_landmarks.part(p2).y)

    # calculate the distance between the points
    dX = x2 - x1
    dY = y2 - y1

    # calculate the angle by radians
    rads = atan2(dY, dX)

    # transform the radians to degrees and return the result
    return degrees(rads)


#
# def right_eye_reg(landmarks):
#     return np.array([(landmarks.part(42).x, landmarks.part(42).y),
#                             (landmarks.part(43).x, landmarks.part(43).y),
#                             (landmarks.part(44).x, landmarks.part(44).y),
#                             (landmarks.part(45).x, landmarks.part(45).y),
#                             (landmarks.part(46).x, landmarks.part(46).y),
#                             (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
#
# def left_eye_reg(landmarks):
#     return np.array([(landmarks.part(36).x, landmarks.part(36).y),
#                             (landmarks.part(37).x, landmarks.part(37).y),
#                             (landmarks.part(38).x, landmarks.part(38).y),
#                             (landmarks.part(39).x, landmarks.part(39).y),
#                             (landmarks.part(40).x, landmarks.part(40).y),
#                             (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

def eye_reg(landmarks, eye_, dict_poitns_eye):



    p1, p2, p3, p4, p5, p6 = dict_poitns_eye[eye_]

    return np.array([(landmarks.part(p2).x, landmarks.part(p2).y),
                            (landmarks.part(p5).x, landmarks.part(p5).y),
                            (landmarks.part(p3).x, landmarks.part(p3).y),
                            (landmarks.part(p1).x, landmarks.part(p1).y),
                            (landmarks.part(p4).x, landmarks.part(p4).y),
                            (landmarks.part(p6).x, landmarks.part(p6).y)], np.int32), p2, p5, p3, p1, p4, p6



def get_distance_points(p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2
    return (p1x - p2x, p1y - p2y)


def get_cycle(thresh, list_of_items):
    dat = pd.Series(list_of_items)
    dat = np.log10(dat).diff(1).dropna()
    cycles_1 = sum(dat > thresh)
    cycles_2 = sum(dat < -1*thresh)
    if (cycles_1 > 1) and (cycles_2 > 1):
        return True
    else:
        return False




### offline
def clean_au_7(AU_7, AU_43E, window_size):
    AU_7_updated = AU_7 - np.multiply(AU_7, AU_43E)
    for i in range(1, window_size + 1):
        AU_7_updated -= np.multiply(AU_7_updated, shift(AU_43E, i, cval = 0))
        AU_7_updated -= np.multiply(AU_7_updated, shift(AU_43E, -i, cval = 0))
    return AU_7_updated



def get_adj_aus(np_all_au, num_hold):
    orig_shape = np_all_au.shape

    np_all_au = np.append(np_all_au, np.zeros([num_hold, np_all_au.shape[1]], np.float64), 0)
    new_np_all_au = np.zeros(np_all_au.shape, np.float64)
    for i in range(num_hold):
        new_np_all_au += np.roll(np_all_au, i, 0)

    new_np_all_au = new_np_all_au[:orig_shape[0], :]
    new_np_all_au = np.where(new_np_all_au >= 1, 1, 0)

    return new_np_all_au


def get_emotions_post1(list_of_all_au_adj, combos_matrix):
    combos_raw = np.zeros([list_of_all_au_adj.shape[0], combos_matrix.shape[1]])

    combos_matrix_np = np.array(combos_matrix, np.float64)


    #  for in range (num of frames)
    for i in range(list_of_all_au_adj.shape[0]):
        # combos_matrix_np = np.where(np.isnan(combos_matrix_np), 0, combos_matrix_np)
        temp_ = np.subtract(combos_matrix_np, list_of_all_au_adj[i].reshape(list_of_all_au_adj.shape[1], 1))
        temp_ = np.where(np.isnan(temp_), 0, temp_)

        temp_ = np.abs(temp_)
        temp_ = np.sum(temp_, 0)

        temp_ = np.where(temp_ == 0, True, False)
        combos_raw[i] = temp_


    combos_raw[:, -8] = correct_AU_57(combos_raw[:, -8])

    combos_raw = np.where(combos_raw == 1, True, False)


    return combos_raw




def get_emotions_post(list_of_all_au_adj, combos_matrix, combo_desc):
    combos = []
    combos_raw = np.zeros([list_of_all_au_adj.shape[0], combos_matrix.shape[1]])
    combo_names = np.array(combos_matrix.columns.tolist())
    combos_matrix_np = np.array(combos_matrix, np.float64)


    for i in range(list_of_all_au_adj.shape[0]):
        temp_ = np.subtract(combos_matrix_np, list_of_all_au_adj[i].reshape(list_of_all_au_adj.shape[1], 1))
        temp_ = np.where(np.isnan(temp_), 0, temp_)
        temp_ = np.abs(temp_)
        temp_ = np.sum(temp_, 0)

        # map(lambda i: combos_raw[i] = np.where(np.sum(np.abs(np.where(
        #     np.isnan(np.subtract(combos_matrix_np, list_of_all_au_adj[i].reshape(list_of_all_au_adj.shape[1], 1))), 0,
        #     temp_)), 0) == 0, True, False), range(list_of_all_au_adj.shape[0]))

        temp_ = np.where(temp_ == 0, True, False)
        combos_raw[i] = temp_

        combos.append([name + " " + combo_desc[name] for name in combo_names[temp_]])
    
    combos_raw[:, -8] = correct_AU_57(combos_raw[:, -8])
    
    combos_raw = np.where(combos_raw == 1, True, False)

    for i in range(list_of_all_au_adj.shape[0]):
        combos.append([name + " " + combo_desc[name] for name in combo_names[combos_raw[i]]])
    
    return combos_raw, combos


def correct_au(to_correct, correct_with, dict_, window_size, au_all):
    to_correct = [dict_[i] for i in to_correct]
    correct_with = [dict_[j] for j in correct_with]

    # print("to correct", to_correct, "correct_with", correct_with)

    for tc in to_correct:
        au_all[:, tc] = clean_au_7(au_all[:, tc], np.any(au_all[:, correct_with], 1) * 1, window_size)
    return au_all



def correct_AU_57(arr):
    out_1 = np.where(pd.Series(arr).rolling(window = 30).sum().fillna(0).values == 20, 1, 0)
    out_2 = pd.Series(out_1[::-1]).rolling(30).sum().fillna(0).values[::-1]
    out_final = np.where(out_2 > 0, 1., 0.)
    return out_final


def clean_speech(list_of_all_au_raw, num_frames_to_clean):

    speech = np.copy(list_of_all_au_raw[:, -1])
    speech = "".join([str(int(i)) for i in speech])

    for i in range(1, num_frames_to_clean):
        pattern_to_find = "0" + "1"*i + "0"
        pattern_to_replace = "0" * (i+2)

        while speech.find(pattern_to_find) != -1:
            speech = speech.replace(pattern_to_find, pattern_to_replace)

    speech = np.array([float(i) for i in speech])

    list_of_all_au_raw[:, -1] = speech

    return list_of_all_au_raw

def get_start_end(temp_, fps):
    start_ = [m.start(0) + 1 for m in re.finditer("01", temp_)]
    end_ = [m.end(0) - 2 for m in re.finditer("10", temp_)]
    if temp_[0] == "1":
        start_ = [0] + start_
    if temp_[-1] == "1":
        end_ = end_ + [len(temp_)]
    return [(i/fps, j/fps) for i, j in zip(start_, end_)]