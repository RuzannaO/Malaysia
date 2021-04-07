import numpy as np
import pandas as pd
import cv2
import dlib
from au.parameters import *
from au.variables import *
from au.au_all_matrix import *
from emotions import emotions_new_matrix as emotions_new
import utils
from scipy import ndimage
from flask import Flask, request, jsonify, Response
from math import hypot
import os
import tensorflow as tf
import datetime
import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from itertools import chain
import json

filename = datetime.datetime.now().timestamp()
filename = str(int(filename))


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
speech_model_name = "1_64_True_True_0.0_lip_motion_net_model.h5"
speech_model = load_model(f"./models/{speech_model_name}")
scaler = MinMaxScaler(feature_range = (0, 1))

font = cv2.FONT_HERSHEY_SIMPLEX

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

graph = tf.get_default_graph()


def post_clean(list_of_all_au_raw):
    list_of_all_au_raw = np.array(list_of_all_au_raw)

    list_of_all_au_raw = clean_speech(list_of_all_au_raw, 3)

    au_all = get_adj_aus(list_of_all_au_raw, 10)

    dict_index = {df.columns.tolist()[j]: j for j in range(df.shape[1])}

    au_all = correct_au(to_correct=["AU_24"], correct_with=["AU_25"], dict_=dict_index, window_size=10, au_all=au_all)
    au_all = correct_au(to_correct=["AU_12_uni"], correct_with=["AU_12_high", "AU_12_low", "AU_12_asym"],
                        dict_=dict_index, window_size=10, au_all=au_all)

    au_all = correct_au(to_correct=["AU_7_left_low",
                                    "AU_7_right_low",
                                    "AU_7_left_high",
                                    "AU_7_right_high",
                                    "AU_7_high",
                                    "AU_7_low",
                                    "AU_7",
                                    "AU_5_left_low",
                                    "AU_5_right_low",
                                    "AU_5_left_high",
                                    "AU_5_right_high",
                                    "AU_5_low",
                                    "AU_5_high",
                                    "AU_5"],
                        correct_with=["CLOSE", "BLINKING", "FAST_BLINKING", "GAZING"],
                        dict_=dict_index, window_size=10, au_all=au_all)

    au_all = correct_au(to_correct=["eye_middle_up", "eye_au_m67", "eye_au_m68", "eye_right_middle"],
                        correct_with=["Combo_49_b"], dict_=dict_index, window_size=10, au_all=au_all)

    combos_raw, comboss = get_emotions_post(au_all, combos_matrix, combo_desc)
    #
    # cap = cv2.VideoCapture(f'./video/{video_name}')
    #
    # num_frames = cap.get(cv2.CAP_PROP_FPS)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # out = cv2.VideoWriter(f'./video_output/{video_name}', fourcc, int(num_frames),
    #                       (int(width), int(height)))

    # ret, frame = cap.read()
    # i = 0
    # while ret:
    #     frame = cv2.putText(img=frame, text=str(combos[i]), org=(100, 100), fontFace=font,
    #                         fontScale=1, color=(255, 255, 255), thickness=2)
    #     out.write(frame)
    #
    #     ret, frame = cap.read()
    #     i += 1
    # cap.release()
    # out.release()

    return comboss


def get_au(COUNTER, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, EYE_AR_THRESH, au_1,
           au_10, au_12_high, au_12_low, au_17, au_24, au_25, au_26, au_2_left, au_2_right, au_55_56_left,
           au_55_56_right, au_59_60_nod, au_59_60_shake, au_5_left_high, au_5_left_low, au_5_right_high, au_5_right_low,
           au_7_left_high, au_7_left_low, au_7_right_high, au_7_right_low, fast_blinking, gazing,
           gray_scale_gabor_au_17, hor_, norm_blinking_freq,
           point_4_parameter, points_10, points_12_left, points_12_middle, points_12_right, points_17, points_24,
           points_25, points_26, points_4, points_4_up, points_left_au_5,
           points_left_au_7, points_right_au_5, points_right_au_7, ratio_1_left, ratio_1_right,
           ratio_2_down, ratio_2_up, ver_, window_parameter, window_parameter_59_60,
           window_parameter_eye, window_parameter_fast_blink_gaze, x_coord, x_coord_thresh, y_coord,
           y_coord_thresh, quick_shift_right, quick_shift_left, au_list_eye_right, au_list_eye_left,window_parameter_speech,speech_points,list_of_au_current_frame, path =None, live=True, frame=None):


    if live:
        list_of_list = []
        frame = np.array(frame)
        ## gets input frame
        try:
            if frame is None:
                return ("invalid frame")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = ndimage.rotate(gray, 180)

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

            x_coord, y_coord, hor_move, ver_move = head_movement(window_parameter, landmarks, x_coord, y_coord,
                                                                 x_coord_thresh, y_coord_thresh)

            ver_, hor_, list_of_au_current_frame = AU_59_60(landmarks, ver_, hor_, window_parameter_59_60, au_59_60_nod,
                                                            au_59_60_shake, list_of_au_current_frame)

            hor_AU_55_56, list_of_au_current_frame, yes = AU_55_56(landmarks, au_55_56_left, au_55_56_right,
                                                                   list_of_au_current_frame)
            if not hor_move or not ver_move:
                if not yes:
                    points_17, points_26, points_4_up, points_4, list_of_au_current_frame = AU_1_2_4(landmarks,
                                                                                                     points_17,
                                                                                                     points_26,
                                                                                                     points_4_up,
                                                                                                     points_4,
                                                                                                     window_parameter,
                                                                                                     au_2_left,
                                                                                                     au_2_right, au_1,
                                                                                                     point_4_parameter,
                                                                                                     list_of_au_current_frame)

                    points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7, list_of_au_current_frame = AU_5_7(
                        landmarks, points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7,
                        window_parameter, au_5_left_low, au_5_right_low, au_5_left_high,
                        au_5_right_high, au_7_left_low, au_7_left_high, au_7_right_low, au_7_right_high,
                        list_of_au_current_frame)

                    points_10, points_12_right, points_12_left, points_12_middle, list_of_au_current_frame = AU_10_12(
                        landmarks, points_10, points_12_right, points_12_left, points_12_middle, window_parameter,
                        au_12_low, au_12_high, au_10, list_of_au_current_frame)

                    gray_scale_gabor_au_17, list_of_au_current_frame = AU_17(gray, landmarks, gray_scale_gabor_au_17,
                                                                             window_parameter, au_17,
                                                                             list_of_au_current_frame)

                    points_24, list_of_au_current_frame = AU_24(landmarks, points_24, window_parameter, au_24,
                                                                list_of_au_current_frame)

                    points_25, points_26, list_of_au_current_frame = AU_25_26(landmarks, points_25, points_26,
                                                                              window_parameter, au_26, au_25,
                                                                              list_of_au_current_frame)

                    COUNTER, fast_blinking, gazing, list_of_au_current_frame = AU_43_45(landmarks, COUNTER,
                                                                                        EYE_AR_THRESH,
                                                                                        EYE_AR_CONSEC_FRAMES_BLINK,
                                                                                        EYE_AR_CONSEC_FRAMES_CLOSE,
                                                                                        fast_blinking, gazing,
                                                                                        window_parameter_fast_blink_gaze,
                                                                                        list_of_au_current_frame,
                                                                                        norm_blinking_freq)

                    au_list_eye_right, list_of_au_current_frame, quick_shift_right = AU_M67_M68_75_76(gray, landmarks,
                                                                                                      au_list_eye_right,
                                                                                                      ratio_1_left,
                                                                                                      ratio_2_up,
                                                                                                      ratio_1_right,
                                                                                                      ratio_2_down,
                                                                                                      window_parameter_eye,
                                                                                                      window_parameter_quick_shift,
                                                                                                      list_of_au_current_frame,
                                                                                                      quick_shift_right,
                                                                                                      "right",
                                                                                                      dict_poitns_eye,
                                                                                                      eye_text)

                    au_list_eye_left, list_of_au_current_frame, quick_shift_left = AU_M67_M68_75_76(gray, landmarks,
                                                                                                    au_list_eye_left,
                                                                                                    ratio_1_left,
                                                                                                    ratio_2_up,
                                                                                                    ratio_1_right,
                                                                                                    ratio_2_down,
                                                                                                    window_parameter_eye,
                                                                                                    window_parameter_quick_shift,
                                                                                                    list_of_au_current_frame,
                                                                                                    quick_shift_left,
                                                                                                    "left",
                                                                                                    dict_poitns_eye,
                                                                                                    eye_text)
                    with graph.as_default():
                        speech_points, list_of_au_current_frame = get_speech(landmarks, window_parameter_speech,
                                                                             list_of_au_current_frame, speech_points,
                                                                             speech_model=speech_model, scaler=scaler)
                # text = [*au_names[np.nonzero(list_of_au_current_frame==1)[0]]]



            list_of_au_current_frame = np.where(np.isnan(list_of_au_current_frame), 0, list_of_au_current_frame)

            # !!!! -11 parameter is mobile
            list_of_au_current_frame[-12: -2] = np.where(list_of_au_current_frame[-12: -2] < 2, 0, 1)
            list_of_list = utils.get_window(10, list_of_list, list_of_au_current_frame)
            list_of_list_np = np.sum(list_of_list, 0)
            list_of_list_np = np.where(list_of_list_np > 1, 1, list_of_list_np)
            combos = emotions_new.get_emotions(list_of_list_np, combos_matrix)
            list_of_list_np = au_names[np.where(list_of_list_np == 1, True, False)].values
            combos = [i + " " + combo_desc[i] for i in combos]




            ### showing the video
            cv2.putText(img=gray, text=str(combos), org=(100, 100), fontFace=font, fontScale=1,
                        color=0, thickness=5)
            cv2.putText(img=gray, text=str(combos), org=(100, 100), fontFace=font, fontScale=1,
                        color=255, thickness=2)

            return {"combos":combos}#,"frame":cv2.imencode('.jpg', gray)[1].tostring()}
        except Exception as e:
            return str(e)

        ### offline
    else:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        print("Calculating combos")
        while ret:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = detector(gray)
                for face in faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()
                    landmarks = predictor(gray, face)

                list_of_au_current_frame = np.zeros(df.shape[0])

                x_coord, y_coord, hor_move, ver_move = head_movement(window_parameter, landmarks, x_coord, y_coord,
                                                                     x_coord_thresh, y_coord_thresh)

                ver_, hor_, list_of_au_current_frame = AU_59_60(landmarks, ver_, hor_, window_parameter_59_60,
                                                                au_59_60_nod, au_59_60_shake,
                                                                list_of_au_current_frame)

                hor_AU_55_56, list_of_au_current_frame, yes = AU_55_56(landmarks, au_55_56_left, au_55_56_right,
                                                                       list_of_au_current_frame)
                if not hor_move or not ver_move:
                    if not yes:
                        points_17, points_26, points_4_up, points_4, list_of_au_current_frame = AU_1_2_4(landmarks,
                                                                                                         points_17,
                                                                                                         points_26,
                                                                                                         points_4_up,
                                                                                                         points_4,
                                                                                                         window_parameter,
                                                                                                         au_2_left,
                                                                                                         au_2_right,
                                                                                                         au_1,
                                                                                                         point_4_parameter,
                                                                                                         list_of_au_current_frame)

                        points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7, list_of_au_current_frame = AU_5_7(
                            landmarks, points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7,
                            window_parameter, au_5_left_low, au_5_right_low, au_5_left_high,
                            au_5_right_high, au_7_left_low, au_7_left_high, au_7_right_low, au_7_right_high,
                            list_of_au_current_frame)

                        points_10, points_12_right, points_12_left, points_12_middle, list_of_au_current_frame = AU_10_12(
                            landmarks, points_10, points_12_right, points_12_left, points_12_middle,
                            window_parameter, au_12_low, au_12_high, au_10, list_of_au_current_frame)

                        gray_scale_gabor_au_17, list_of_au_current_frame = AU_17(gray, landmarks,
                                                                                 gray_scale_gabor_au_17,
                                                                                 window_parameter, au_17,
                                                                                 list_of_au_current_frame)

                        points_24, list_of_au_current_frame = AU_24(landmarks, points_24, window_parameter, au_24,
                                                                    list_of_au_current_frame)

                        points_25, points_26, list_of_au_current_frame = AU_25_26(landmarks, points_25, points_26,
                                                                                  window_parameter, au_26, au_25,
                                                                                  list_of_au_current_frame)

                        COUNTER, fast_blinking, gazing, list_of_au_current_frame = AU_43_45(landmarks, COUNTER,
                                                                                            EYE_AR_THRESH,
                                                                                            EYE_AR_CONSEC_FRAMES_BLINK,
                                                                                            EYE_AR_CONSEC_FRAMES_CLOSE,
                                                                                            fast_blinking, gazing,
                                                                                            window_parameter_fast_blink_gaze,
                                                                                            list_of_au_current_frame,
                                                                                            norm_blinking_freq)

                        au_list_eye_right, list_of_au_current_frame, quick_shift_right = AU_M67_M68_75_76(gray,
                                                                                                          landmarks,
                                                                                                          au_list_eye_right,
                                                                                                          ratio_1_left,
                                                                                                          ratio_2_up,
                                                                                                          ratio_1_right,
                                                                                                          ratio_2_down,
                                                                                                          window_parameter_eye,
                                                                                                          window_parameter_quick_shift,
                                                                                                          list_of_au_current_frame,
                                                                                                          quick_shift_right,
                                                                                                          "right",
                                                                                                          dict_poitns_eye,
                                                                                                          eye_text)
                        au_list_eye_left, list_of_au_current_frame, quick_shift_left = AU_M67_M68_75_76(gray,
                                                                                                        landmarks,
                                                                                                        au_list_eye_left,
                                                                                                        ratio_1_left,
                                                                                                        ratio_2_up,
                                                                                                        ratio_1_right,
                                                                                                        ratio_2_down,
                                                                                                        window_parameter_eye,
                                                                                                        window_parameter_quick_shift,
                                                                                                        list_of_au_current_frame,
                                                                                                        quick_shift_left,
                                                                                                        "left",
                                                                                                        dict_poitns_eye,
                                                                                                        eye_text)
                        speech_points, list_of_au_current_frame = get_speech(landmarks, window_parameter_speech,
                                                                             list_of_au_current_frame,
                                                                             speech_points,
                                                                             speech_model=speech_model,
                                                                             scaler=scaler)

                # !!!! -11 parameter is mobile
                list_of_au_current_frame[-12: -2] = np.where(list_of_au_current_frame[-12: -2] < 2, 0, 1)
                list_of_au_current_frame = np.where(np.isnan(list_of_au_current_frame), 0, list_of_au_current_frame)

                list_of_all_au_raw.append(list_of_au_current_frame)

                key = cv2.waitKey(1)
                if key == 27:
                    break

                ret, frame = cap.read()

            except NameError:
                # print(1)
                continue
            except RuntimeWarning:
                # print(2)
                continue
            except Exception as e:
                # print(e)
                continue
        cap.release()
        cv2.destroyAllWindows()

        list_of_au_current_frame[-12: -2] = np.where(list_of_au_current_frame[-12: -2] < 2, 0, 1)
        list_of_au_current_frame = np.where(np.isnan(list_of_au_current_frame), 0, list_of_au_current_frame)

        list_of_all_au_raw.append(list_of_au_current_frame)
        print("Creaning the combos")
        combos = post_clean(list_of_all_au_raw)
        print("Done")
        return {'offline':combos}


# a= []
# cap = cv2.VideoCapture(0)
# while True:
#     try:
#         _, frame = cap.read()
#         r = get_au(COUNTER, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, EYE_AR_THRESH, au_1,
#            au_10, au_12_high, au_12_low, au_17, au_24, au_25, au_26, au_2_left, au_2_right, au_55_56_left,
#            au_55_56_right, au_59_60_nod, au_59_60_shake, au_5_left_high, au_5_left_low, au_5_right_high, au_5_right_low,
#            au_7_left_high, au_7_left_low, au_7_right_high, au_7_right_low, fast_blinking, gazing,
#            gray_scale_gabor_au_17, hor_, norm_blinking_freq,
#            point_4_parameter, points_10, points_12_left, points_12_middle, points_12_right, points_17, points_24,
#            points_25, points_26, points_4, points_4_up, points_left_au_5,
#            points_left_au_7, points_right_au_5, points_right_au_7, ratio_1_left, ratio_1_right,
#            ratio_2_down, ratio_2_up, ver_, window_parameter, window_parameter_59_60,
#            window_parameter_eye, window_parameter_fast_blink_gaze, x_coord, x_coord_thresh, y_coord,
#            y_coord_thresh, quick_shift_right, quick_shift_left, au_list_eye_right, au_list_eye_left, window_parameter_speech, speech_points,list_of_au_current_frame, live = True, frame=frame)
#         a.append(r)
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1)
#         if key == 27:
#             break
#     except KeyboardInterrupt:
#         break
#     # except NameError as e:
#     #     print(1, e)
#     #     continue
#     # except RuntimeWarning as e:
#     #     print(2, e)
#     #     continue
#     # except Exception as e:
#     #     print(e)
#     #     continue
# cap.release()
# cv2.destroyAllWindows()




app = Flask(__name__)
@app.route('/offline', methods=['GET', 'POST'])
def vedeo_au():
    input_json = json.loads(request.data.decode('utf-8'))
    print(input_json)
    path = input_json["path"]
    print(path)
    return jsonify(get_au(COUNTER, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, EYE_AR_THRESH, au_1,
           au_10, au_12_high, au_12_low, au_17, au_24, au_25, au_26, au_2_left, au_2_right, au_55_56_left,
           au_55_56_right, au_59_60_nod, au_59_60_shake, au_5_left_high, au_5_left_low, au_5_right_high, au_5_right_low,
           au_7_left_high, au_7_left_low, au_7_right_high, au_7_right_low, fast_blinking, gazing,
           gray_scale_gabor_au_17, hor_, norm_blinking_freq,
           point_4_parameter, points_10, points_12_left, points_12_middle, points_12_right, points_17, points_24,
           points_25, points_26, points_4, points_4_up, points_left_au_5,
           points_left_au_7, points_right_au_5, points_right_au_7, ratio_1_left, ratio_1_right,
           ratio_2_down, ratio_2_up, ver_, window_parameter, window_parameter_59_60,
           window_parameter_eye, window_parameter_fast_blink_gaze, x_coord, x_coord_thresh, y_coord,
           y_coord_thresh, quick_shift_right, quick_shift_left, au_list_eye_right, au_list_eye_left, window_parameter_speech, speech_points,list_of_au_current_frame, path = path, live = False))


@app.route('/live', methods=['GET', 'POST'])
def video_feed():
    img = request.data
    buff = np.fromstring(img, np.uint8)
    buff = buff.reshape(1, -1)
    frame = cv2.imdecode(buff, cv2.IMREAD_COLOR)

    return jsonify(get_au(COUNTER, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, EYE_AR_THRESH, au_1,
           au_10, au_12_high, au_12_low, au_17, au_24, au_25, au_26, au_2_left, au_2_right, au_55_56_left,
           au_55_56_right, au_59_60_nod, au_59_60_shake, au_5_left_high, au_5_left_low, au_5_right_high, au_5_right_low,
           au_7_left_high, au_7_left_low, au_7_right_high, au_7_right_low, fast_blinking, gazing,
           gray_scale_gabor_au_17, hor_, norm_blinking_freq,
           point_4_parameter, points_10, points_12_left, points_12_middle, points_12_right, points_17, points_24,
           points_25, points_26, points_4, points_4_up, points_left_au_5,
           points_left_au_7, points_right_au_5, points_right_au_7, ratio_1_left, ratio_1_right,
           ratio_2_down, ratio_2_up, ver_, window_parameter, window_parameter_59_60,
           window_parameter_eye, window_parameter_fast_blink_gaze, x_coord, x_coord_thresh, y_coord,
           y_coord_thresh, quick_shift_right, quick_shift_left, au_list_eye_right, au_list_eye_left, window_parameter_speech, speech_points,list_of_au_current_frame, live = True, frame=frame))



if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000, use_reloader=False)
