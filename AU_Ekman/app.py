import numpy as np
import pandas as pd
from math import hypot

import os
import sys
import time
import datetime

import keras
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage

import cv2
import dlib

from itertools import chain
import logging

import utils
from au.variables import *
from au.parameters import *
from au.au_all_matrix import *
from emotions import emotions_new_matrix as emotions_new

filename = datetime.datetime.now().timestamp()
filename = str(int(filename))

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s %(message)s')
#
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)
#
# handler_1 = logging.FileHandler("./log_files/{}.log".format(filename),"w", encoding=None, delay="true")
# handler_1.setLevel(logging.INFO)
# handler_1.setFormatter(formatter)
#
# logger.addHandler(handler)
# logger.addHandler(handler_1)




# video_name = sys.argv[1]
# video_name = "speech.mov"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

speech_model_name = "1_64_True_True_0.0_lip_motion_net_model.h5"
speech_model = load_model(f"./models/{speech_model_name}")
scaler = MinMaxScaler(feature_range = (0, 1))


font = cv2.FONT_HERSHEY_SIMPLEX


# if not os.path.isfile(f"./video/{video_name}"):
#     print("video not found ¯\_(ツ)_/¯")
#     sys.exit()



df = pd.read_csv("matrix.csv")
dict_ = {i: j for i, j in zip(df.columns, df.values)}

combos_matrix = pd.read_csv("combos_new.csv")
combos_matrix = pd.DataFrame(combos_matrix.T)
combos_matrix.columns = combos_matrix.iloc[0]
au_names = pd.Series(combos_matrix.index.tolist()[1:])
combos_matrix = combos_matrix.reset_index()
combos_matrix = combos_matrix.drop(["index"], axis=1)
combos_matrix = combos_matrix.drop([0], axis=0)

combo_desc = pd.read_csv("combo_desc.csv")
combo_desc = {j: i for i, j in combo_desc.values}






# cap = cv2.VideoCapture(f"./video/{video_name}")
# ret, frame = cap.read()

cap = cv2.VideoCapture(0)
while True:
    try:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = ndimage.rotate(gray, 180)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            landmarks = predictor(gray, face)
            for i in range(68):
                cv2.circle(gray, (landmarks.part(i).x, landmarks.part(i).y), 5, (0, 0, 255), -1)
            break
        list_of_au_current_frame = np.zeros(df.shape[0])

        x_coord, y_coord, hor_move, ver_move = head_movement(window_parameter, landmarks, x_coord, y_coord, x_coord_thresh, y_coord_thresh)

        ver_, hor_, list_of_au_current_frame = AU_59_60(landmarks, ver_, hor_, window_parameter_59_60, au_59_60_nod, au_59_60_shake, list_of_au_current_frame)

        hor_AU_55_56, list_of_au_current_frame, yes = AU_55_56(landmarks, au_55_56_left, au_55_56_right, list_of_au_current_frame)
        if not hor_move or not ver_move:
            if not yes:
                points_17, points_26, points_4_up, points_4, list_of_au_current_frame = AU_1_2_4(landmarks, points_17, points_26, points_4_up, points_4, window_parameter, au_2_left, au_2_right, au_1, point_4_parameter, list_of_au_current_frame)

                points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7, list_of_au_current_frame = AU_5_7(landmarks, points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7, window_parameter, au_5_left_low, au_5_right_low, au_5_left_high,
                                                                                                                          au_5_right_high, au_7_left_low, au_7_left_high, au_7_right_low, au_7_right_high, list_of_au_current_frame)

                points_10, points_12_right, points_12_left, points_12_middle, list_of_au_current_frame = AU_10_12(landmarks, points_10, points_12_right, points_12_left, points_12_middle, window_parameter, au_12_low, au_12_high, au_10, list_of_au_current_frame)

                gray_scale_gabor_au_17, list_of_au_current_frame = AU_17(gray, landmarks, gray_scale_gabor_au_17, window_parameter, au_17, list_of_au_current_frame)

                points_24, list_of_au_current_frame = AU_24(landmarks, points_24, window_parameter, au_24, list_of_au_current_frame)

                points_25, points_26, list_of_au_current_frame = AU_25_26(landmarks, points_25, points_26, window_parameter, au_26, au_25, list_of_au_current_frame)

                # gray_scale_gabor_nose_wrinkel, list_of_au_current_frame,filtered_img = nose_wrinkle(gray, landmarks, nose_wrinkle_thresh, gray_scale_gabor_nose_wrinkel, window_parameter, list_of_au_current_frame)

                COUNTER, fast_blinking, gazing, list_of_au_current_frame = AU_43_45(landmarks, COUNTER, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, fast_blinking, gazing, window_parameter_fast_blink_gaze, list_of_au_current_frame, norm_blinking_freq)


                au_list_eye_right, list_of_au_current_frame, quick_shift_right, points_area_right = AU_M67_M68_75_76(gray, landmarks, au_list_eye_right, ratio_1_left, ratio_2_up, ratio_1_right, ratio_2_down,
                                                                                                window_parameter_eye,
                                                                                                window_parameter_quick_shift,
                                                                                                list_of_au_current_frame,
                                                                                                quick_shift_right,
                                                                                                "right", dict_poitns_eye,
                                                                                                eye_text, points_area_right, window_parameter, area_parameter)
                au_list_eye_left, list_of_au_current_frame, quick_shift_left, points_area_left = AU_M67_M68_75_76(gray, landmarks, au_list_eye_left, ratio_1_left, ratio_2_up, ratio_1_right, ratio_2_down,
                                                                                              window_parameter_eye,
                                                                                              window_parameter_quick_shift,
                                                                                              list_of_au_current_frame,
                                                                                              quick_shift_left,
                                                                                              "left", dict_poitns_eye,
                                                                                              eye_text, points_area_left, window_parameter, area_parameter)
                speech_points, list_of_au_current_frame = get_speech(landmarks, window_parameter_speech, list_of_au_current_frame, speech_points, speech_model = speech_model, scaler = scaler)
        
        list_of_au_current_frame = np.where(np.isnan(list_of_au_current_frame), 0, list_of_au_current_frame)

        # !!!! -11 parameter is mobile
        list_of_au_current_frame[-12: -2] = np.where(list_of_au_current_frame[-12: -2] < 2, 0, 1)
        list_of_list = utils.get_window(10, list_of_list, list_of_au_current_frame)
        list_of_list_np = np.sum(list_of_list, 0)
        list_of_list_np = np.where(list_of_list_np > 1, 1, list_of_list_np)
        combos = emotions_new.get_emotions(list_of_list_np, combos_matrix)
        list_of_list_np = au_names[np.where(list_of_list_np == 1, True, False)].values
        combos = [i + " " + combo_desc[i] for i in combos]

        text = [*au_names[np.nonzero(list_of_au_current_frame >= 1)[0]]]
        print(text)

        cv2.putText(img=gray, text=str(combos), org=(100, 100), fontFace=font, fontScale=1,
                    color=0, thickness=5)
        cv2.putText(img = gray, text = str(combos), org = (100,100), fontFace = font, fontScale = 1,
                color =255 ,thickness=2)

        # logging.info(combos)
        # logging.info(list_of_list_np)
        # logging.info("\n")
        cv2.imshow("Frame", gray)
        key = cv2.waitKey(1)
        if key == 27:
            break
    except KeyboardInterrupt:
        break
    except NameError as e:
        print(1, e)
        continue
    except RuntimeWarning as e:
        print(2, e)
        continue
    # except Exception as e:
    #   print(e)
    #   continue
cap.release()
cv2.destroyAllWindows()