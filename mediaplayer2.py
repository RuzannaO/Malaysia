#Gotcha version 2.0
import PySimpleGUI as sg
import cv2
import signal
import pyaudio
from collections import OrderedDict
from shutil import copyfile
from av_recording import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from qtimeline import VideoSample
from MainWindow import Ui_MainWindow

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing import concatenate

# from emotion_bar import configureChart
from datetime import datetime
import numpy as np
import pandas as pd
from math import hypot
import os
import shutil
import sys
import time

from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage

import dlib
import json

from itertools import chain
import logging

from AU_Ekman import utils
from AU_Ekman.au.variables import *
from AU_Ekman.au.parameters import *

from AU_Ekman.au.au_all_matrix import *
from AU_Ekman.emotions import emotions_new_matrix as emotions_new
import keras
from keras.models import load_model


filename = datetime.now().timestamp()
filename = str(int(filename))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

speech_model_name = "1_64_True_True_0.0_lip_motion_net_model.h5"
speech_model = load_model(f"./models/{speech_model_name}")
scaler = MinMaxScaler(feature_range=(0, 1))
from pathlib import Path

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

def camera_available():
    cameras = QCameraInfo.availableCameras()
    if len(cameras) == 0:
        return False
    else:
        return True


def audio_available():
    audio = QAudioDeviceInfo.availableDevices(QAudio.AudioOutput)
    if len(audio) == 0:
        return False
    else:
        return True
# sortes emotion durations and removes empty ones
def sorted_list(dict_init):
    sorted_lst = []
    newdict = {}
    for j in dict_init.keys():
        em_dur = 0
        for i in dict_init[j]:
            em_dur +=  i[1]-i[0]
        if em_dur != 0:
            newdict[j] = em_dur
    sorted_lst = list(OrderedDict(sorted(newdict.items(), key=lambda t: t[1])))[::-1]
    return(sorted_lst)




class Convert(QThread):

    def __init__(self, parent):
        QThread.__init__(self, parent)
        self.stopVideo = False
        self.width = 640
        self.height = 480
        self.output_dir = "recordings" + os.path.sep
        now = datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.splash = None
        self.cap = None
        self.filename = None
        self.resultpath = None
        self.emotionDict = None


    def run(self):
        # reload_it()


        global landmarks, filename, df, dict_, combos_matrix, combo_desc, au_names, font, scaler, predictor, detector, COUNTER, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, EYE_AR_THRESH, In, Out, _, _10, _11, _13, _14, _16, _17, _19, _23, _3, _4, _5, _7, _9, ___, __builtin__, __builtins__, __doc__, __loader__, __name__, __package__, __spec__, _dh, _i, _i1, _i10, _i11, _i12, _i13, _i14, _i15, _i16, _i17, _i18, _i19, _i2, _i20, _i21, _i22, _i23, _i24, _i3, _i4, _i5, _i6, _i7, _i8, _i9, _ih, _ii, _iii, _oh, au_1, au_10, au_12_high, au_12_low, au_17, au_21_22_parameter, au_24, au_25, au_26, au_2_left, au_2_right, au_55_56_left, au_55_56_right, au_59_60_nod, au_59_60_shake, au_5_left_high, au_5_left_low, au_5_right_high, au_5_right_low, au_7_left_high, au_7_left_low, au_7_right_high, au_7_right_low, au_list_eye_left, au_list_eye_right, au_list_left, au_list_right, bottomLeftCornerOfText, camera_parameter, cv2, df, dict_, dict_emotion, dict_poitns_eye, dlib, eye_text, eye_thresh, fast_blinking, font, fontColor, fontScale, forehead_wrinkle, gazing, gray_scale_gabor_au_17, gray_scale_gabor_forehead_wrinkle, gray_scale_gabor_nose_wrinkel, hor_, hor_AU_55_56, lineType, list_of_all_au_raw, list_of_au_current_frame, list_of_list, norm_blinking_freq, nose_wrinkle_thresh, np, pd, point_4_parameter, points_10, points_12_left, points_12_middle, points_12_right, points_17, points_21, points_21_22, points_22, points_24, points_25, points_25_26, points_26, points_4, points_4_gabor, points_4_gabor_parameter, points_4_up, points_left_AU14, points_left_au_5, points_left_au_7, points_left_au_7_low, points_right_AU14, points_right_au_5, points_right_au_7, points_right_au_7_low, quick_shift_left, quick_shift_right, ratio_1_left, ratio_1_right, ratio_2_down, ratio_2_up, re, speech_points, time, utils, ver_, window_parameter, window_parameter_59_60, window_parameter_eye, window_parameter_fast_blink_gaze, window_parameter_quick_shift, window_parameter_speech, points_area_right,points_area_left, window_parameter, area_parameter, x_coord, x_coord_thresh, y_coord, y_coord_thresh

        # makes up an audio file
        video = VideoFileClip(self.filename)
        try:
            vid = video.without_audio()
            audioclip = AudioFileClip(self.filename)
            audioclip.write_audiofile(self.filename[:-3] + "wav")
        except:
            # self.filename = None
            self.splash.close()
            sg.Popup("Incorrect file. Application will close now.")

        self.t1 = time.time()

        p = Path(self.filename)
        self.stem = p.stem + p.suffix
        self.stem_json = p.stem +".json"

        scaler = MinMaxScaler(feature_range=(0, 1))

        df = pd.read_csv("matrix.csv")
        dict_ = {i: j for i, j in zip(df.columns, df.values)}


        combos_matrix = pd.read_csv("combos_new.csv")
        combos_matrix = pd.DataFrame(combos_matrix.T)
        combos_matrix.columns = combos_matrix.iloc[0]
        au_names = pd.Series(combos_matrix.index.tolist()[1:])
        combos_matrix = combos_matrix.reset_index()
        combos_matrix = combos_matrix.drop(["index"], axis=1)
        combos_matrix = combos_matrix.drop([0], axis=0)

        combos_series = pd.Series(combos_matrix.columns.tolist())

        combo_desc = pd.read_csv("combo_desc.csv")
        combo_desc = {j: i for i, j in combo_desc.values}
        cap = cv2.VideoCapture(self.filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        ret, frame = cap.read()

        x_coord = []
        y_coord = []
        list_of_all_au_raw = []

        # input_parameters = [ver_, hor_, points_17, points_26, points_4_up, points_4, points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7,points_10, points_12_right, points_12_left, points_12_middle,gray_scale_gabor_au_17, points_24, points_25, points_26, fast_blinking, gazing, au_list_eye_right, quick_shift_right, points_area_right, au_list_eye_left, quick_shift_left, points_area_left, speech_points]
        ver_, hor_, points_17, points_26, points_4_up, points_4, points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7,points_10, points_12_right, points_12_left, points_12_middle,gray_scale_gabor_au_17, points_24, points_25, points_26, fast_blinking, gazing, au_list_eye_right, quick_shift_right, points_area_right, au_list_eye_left, quick_shift_left, points_area_left, speech_points = ([] for _ in range(27))

        list_of_au_current_frame = np.zeros(df.shape[0])

        # sets up a default rectangle to enable predicting landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dlibRect = dlib.rectangle(10, 50, 100, 200)
        landmarks = predictor(gray, dlibRect)

        # print(x_coord, y_coord, "****", x_coord_thresh, y_coord_thresh)

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

                # print(f'{time.time()-self.t1}')

                x_coord, y_coord, hor_move, ver_move = head_movement(window_parameter, landmarks, x_coord, y_coord,
                                                                     x_coord_thresh, y_coord_thresh)


                print("---processing----")

                ver_, hor_, list_of_au_current_frame = AU_59_60(landmarks, ver_, hor_, window_parameter_59_60,
                                                                au_59_60_nod, au_59_60_shake, list_of_au_current_frame)

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
                            landmarks, points_10, points_12_right, points_12_left, points_12_middle, window_parameter,
                            au_12_low, au_12_high, au_10, list_of_au_current_frame)

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

                        au_list_eye_right, list_of_au_current_frame, quick_shift_right, points_area_right = AU_M67_M68_75_76(
                            gray, landmarks, au_list_eye_right, ratio_1_left, ratio_2_up, ratio_1_right, ratio_2_down,
                            window_parameter_eye,
                            window_parameter_quick_shift,
                            list_of_au_current_frame,
                            quick_shift_right,
                            "right", dict_poitns_eye,
                            eye_text, points_area_right, window_parameter, area_parameter)
                        au_list_eye_left, list_of_au_current_frame, quick_shift_left, points_area_left = AU_M67_M68_75_76(
                            gray, landmarks, au_list_eye_left, ratio_1_left, ratio_2_up, ratio_1_right, ratio_2_down,
                            window_parameter_eye,
                            window_parameter_quick_shift,
                            list_of_au_current_frame,
                            quick_shift_left,
                            "left", dict_poitns_eye,
                            eye_text, points_area_left, window_parameter, area_parameter)
                        speech_points, list_of_au_current_frame = get_speech(landmarks, window_parameter_speech,
                                                                             list_of_au_current_frame, speech_points,
                                                                             speech_model=speech_model, scaler=scaler)

                # !!!! -11 parameter is mobile
                list_of_au_current_frame[-12: -2] = np.where(list_of_au_current_frame[-12: -2] < 2, 0, 1)
                list_of_au_current_frame = np.where(np.isnan(list_of_au_current_frame), 0, list_of_au_current_frame)


                list_of_all_au_raw.append(list_of_au_current_frame)

                key = cv2.waitKey(1)
                if key == 27:
                    break

                ret, frame = cap.read()

            except KeyboardInterrupt:
                break
            except NameError:
                print("NameError: ",1)
                continue
            except RuntimeWarning:
                print("RuntimeWarning",2)
                continue
            except Exception as e:
                print("Exception",e)
                sg.Popup("Corrupt file. Will close now.")
                exit(1)
        cap.release()
        cv2.destroyAllWindows()

        list_of_all_au_raw = np.array(list_of_all_au_raw)

        list_of_all_au_raw = clean_speech(list_of_all_au_raw, 3)

        au_all = get_adj_aus(list_of_all_au_raw, 10)

        dict_index = {df.columns.tolist()[j]: j for j in range(df.shape[1])}

        au_all = correct_au(to_correct=["AU_24"], correct_with=["AU_25"], dict_=dict_index, window_size=10,
                            au_all=au_all)
        au_all = correct_au(to_correct=["AU_12_uni"], correct_with=["AU_12_high", "AU_12_low", "AU_12_asym"],
                            dict_=dict_index, window_size=10, au_all=au_all)

        au_all = correct_au(to_correct=["AU_5_left_low",
                                        "AU_5_right_low",
                                        "AU_5_left_high",
                                        "AU_5_right_high",
                                        "AU_5_low",
                                        "AU_5_high",
                                        "AU_5"],
                            correct_with=["CLOSE", "BLINKING", "FAST_BLINKING"],
                            dict_=dict_index, window_size=20, au_all=au_all)

        au_all = correct_au(to_correct=["AU_7_left_low",
                                        "AU_7_right_low",
                                        "AU_7_left_high",
                                        "AU_7_right_high",
                                        "AU_7_high",
                                        "AU_7_low",
                                        "AU_7"],
                            correct_with=["BLINKING","FAST_BLINKING", "GAZING"],
                            dict_=dict_index, window_size=10, au_all=au_all)


        au_all = correct_au(to_correct=["eye_middle_up", "eye_au_m67", "eye_au_m68", "eye_right_middle"],
                            correct_with=["Combo_49_b"], dict_=dict_index, window_size=10, au_all=au_all)

        combos_raw, combos = get_emotions_post(au_all, combos_matrix, combo_desc)
        start_end_dict = {}

        for emotion in dict_emotion.keys():
            temp_ = np.any(combos_raw[:, np.where(combos_series.isin(dict_emotion[emotion]))[0]], 1) * 1
            temp_ = [str(i) for i in temp_]
            temp_ = "".join(temp_)
            start_end_dict[emotion] = get_start_end(temp_, fps)


        p = Path(self.filename)

        p = p.parent / (p.stem + "_out" + p.suffix)
        p_json = p.parent / (str(p.stem)[:-3]+ "_final" + p.suffix)

        with open(p_json.with_suffix('.json'), "w") as f:
            f.write(json.dumps(start_end_dict))

        cap = cv2.VideoCapture(self.filename)

        num_frames = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'m1v')
        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        out = cv2.VideoWriter(str(p.with_suffix('.mp4')), fourcc, int(num_frames),
                              (int(width), int(height)))

        i = 0
        ret, frame = cap.read()
        while ret:
            # frame = cv2.putText(img=frame, text=str(combos[i]), org=(100, 100), fontFace=font,
            #                     fontScale=1, color=(255, 255, 255), thickness=2)
            out.write(frame)

            ret, frame = cap.read()
            i += 1
        cap.release()
        out.release()

        # the following part form older version Dec/07//2020
        # oldVideoPath =  Path(self.filename)
        # oldPathMoved = "./recordings"+str(("/OriginalVideos/"+oldVideoPath.stem  + oldVideoPath.suffix))
        # shutil.move(str(oldVideoPath), oldPathMoved)
        # self.resultpath = p.with_suffix('.mp4')
        # self.emotionDict  = start_end_dict



        local_path = os.getcwd()
        source = self.filename

        target =f'./recordings/OriginalVideos/{self.stem}'



        # adding exception handling
        # local_path = os.getcwd()
        if os.path.exists(self.filename):

            try:
                copyfile(source, target)
            except IOError as e:
                print("Unable to copy file. %s" % e)
                self.splash.close()
                # sg.Popup("Unable to copy file to recordings\OriginalVideos folder.  Please, select a file from another folder")
                pass
                # exit(1)
            except:
                print("Unexpected error:", sys.exc_info())
                exit(1)


        p = p.parent / (p.stem[:-4] + "__final" + p.suffix)

        self.resultpath = p.with_suffix('.mp4')


        self.emotionDict  = start_end_dict
        # print(os.path.exists(p.with_suffix('.txt')))
        # with open("./recordings/testfile_offline.txt", "w+") as f:
        #     f.write(str(fps))
        #     print("fps_cought")
        video_processed = VideoFileClip(f'{str(p)[:-10]}out.mp4')

        final = concatenate.concatenate_videoclips([video_processed]).set_audio(audioclip)
        final.write_videofile(f'{str(p)[:-10]}_final.mp4')
        video_processed.close()
        audioclip.close()




        # The process cannot access the file because it is being used by another process:
        # if os.path.exists(str(local_path)+"/recordings/" + str(self.stem)):
        #     os.remove(str(local_path)+"/recordings/"+ str(self.stem))

        print(time.time()-self.t1)

        remove_file(f'{str(p)[:-10]}out.mp4')
        remove_file(f'{str(p)[:-11]}.wav')

        return



    def stop(self):
        self.splash.close()

        self.stopVideo = True
        self.cap.release()
        self.sleep(1)


class Thread1(QThread):
    changePixmap = pyqtSignal(QImage)
    webCameraStatusChanged = pyqtSignal(bool)
    def __init__(self, parent,camDeviceId):
        QThread.__init__(self, parent)
        self.cameraDeviceId = camDeviceId
        if not camera_available():
            sg.Popup("No camera")
            self.stop()


    def run(self):
        camId = self.cameraDeviceId
        filename = "./recordings" + os.path.sep + datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.av_rec = start_AVrecording(camId,filename)
        self.av_rec.start(camId,filename)
        return


    def stop(self):
        if camera_available():
            self.av_rec.stop()
            # self.sleep(1)
            file_manager()
            sg.Popup(' Done! ')
        return


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    webCameraStatusChanged = pyqtSignal(bool)
    def __init__(self, parent, camDeviceId):
        QThread.__init__(self, parent)
        self.stopVideo = False
        self.width = 640
        self.height = 480
        self.output_dir = "recordings" + os.path.sep

        self.now = datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.cameraDeviceId = camDeviceId
        print(" self.cameraDeviceId ", self.cameraDeviceId)
        self.cap = cv2.VideoCapture(self.cameraDeviceId)
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.out = cv2.VideoWriter(self.output_dir + self.now + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                   (self.width, self.height))

        if not camera_available():
            sg.Popup("No camera")
            self.stopVideo = True
            self.out.release()


    def run(self):

        # if self.cap.read()[0]==False:
        #     self.cap.release()
        #     self.out.release()
        #     self.webCameraStatusChanged.emit(True)
        #     return

        global landmarks, filename, df, dict_, combos_matrix, combo_desc, au_names, font, scaler, predictor, detector, COUNTER, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, EYE_AR_THRESH, In, Out, _, _10, _11, _13, _14, _16, _17, _19, _23, _3, _4, _5, _7, _9, ___, __builtin__, __builtins__, __doc__, __loader__, __name__, __package__, __spec__, _dh, _i, _i1, _i10, _i11, _i12, _i13, _i14, _i15, _i16, _i17, _i18, _i19, _i2, _i20, _i21, _i22, _i23, _i24, _i3, _i4, _i5, _i6, _i7, _i8, _i9, _ih, _ii, _iii, _oh, au_1, au_10, au_12_high, au_12_low, au_17, au_21_22_parameter, au_24, au_25, au_26, au_2_left, au_2_right, au_55_56_left, au_55_56_right, au_59_60_nod, au_59_60_shake, au_5_left_high, au_5_left_low, au_5_right_high, au_5_right_low, au_7_left_high, au_7_left_low, au_7_right_high, au_7_right_low, au_list_eye_left, au_list_eye_right, au_list_left, au_list_right, bottomLeftCornerOfText, camera_parameter, cv2, df, dict_, dict_emotion, dict_poitns_eye, dlib, eye_text, eye_thresh, fast_blinking, font, fontColor, fontScale, forehead_wrinkle, gazing, gray_scale_gabor_au_17, gray_scale_gabor_forehead_wrinkle, gray_scale_gabor_nose_wrinkel, hor_, hor_AU_55_56, lineType, list_of_all_au_raw, list_of_au_current_frame, list_of_list, norm_blinking_freq, nose_wrinkle_thresh, np, pd, point_4_parameter, points_10, points_12_left, points_12_middle, points_12_right, points_17, points_21, points_21_22, points_22, points_24, points_25, points_25_26, points_26, points_4, points_4_gabor, points_4_gabor_parameter, points_4_up, points_left_AU14, points_left_au_5, points_left_au_7, points_left_au_7_low, points_right_AU14, points_right_au_5, points_right_au_7, points_right_au_7_low, quick_shift_left, quick_shift_right, ratio_1_left, ratio_1_right, ratio_2_down, ratio_2_up, re, speech_points, time, utils, ver_, window_parameter, window_parameter_59_60, window_parameter_eye, window_parameter_fast_blink_gaze, window_parameter_quick_shift, window_parameter_speech, points_area_right, points_area_left, window_parameter, area_parameter, x_coord, x_coord_thresh, y_coord, y_coord_thresh

        # cap = cv2.VideoCapture(f"./video/{video_name}")
        # ret, frame = cap.read()
        self.t1 = time.time()
        self.frames = 0

        while self.stopVideo == False:
            if 1:
                _, frame = self.cap.read()
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
                if len(faces) != 0:
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

                            # gray_scale_gabor_nose_wrinkel, list_of_au_current_frame,filtered_img = nose_wrinkle(gray, landmarks, nose_wrinkle_thresh, gray_scale_gabor_nose_wrinkel, window_parameter, list_of_au_current_frame)

                            COUNTER, fast_blinking, gazing, list_of_au_current_frame = AU_43_45(landmarks, COUNTER,
                                                                                                EYE_AR_THRESH,
                                                                                                EYE_AR_CONSEC_FRAMES_BLINK,
                                                                                                EYE_AR_CONSEC_FRAMES_CLOSE,
                                                                                                fast_blinking, gazing,
                                                                                                window_parameter_fast_blink_gaze,
                                                                                                list_of_au_current_frame,
                                                                                                norm_blinking_freq)

                            au_list_eye_right, list_of_au_current_frame, quick_shift_right, points_area_right = AU_M67_M68_75_76(
                                gray,
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
                                eye_text,
                                points_area_right,
                                window_parameter,
                                area_parameter)
                            au_list_eye_left, list_of_au_current_frame, quick_shift_left, points_area_left = AU_M67_M68_75_76(
                                gray,
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
                                eye_text,
                                points_area_left,
                                window_parameter,
                                area_parameter)
                            speech_points, list_of_au_current_frame = get_speech(landmarks, window_parameter_speech,
                                                                                 list_of_au_current_frame,
                                                                                 speech_points,
                                                                                 speech_model=speech_model,
                                                                                 scaler=scaler)

                list_of_au_current_frame = np.where(np.isnan(list_of_au_current_frame), 0, list_of_au_current_frame)

                # !!!! -11 parameter is mobile
                list_of_au_current_frame[-12: -2] = np.where(list_of_au_current_frame[-12: -2] < 2, 0, 1)
                list_of_list = utils.get_window(10, list_of_list, list_of_au_current_frame)
                list_of_list_np = np.sum(list_of_list, 0)
                list_of_list_np = np.where(list_of_list_np > 1, 1, list_of_list_np)
                combos = emotions_new.get_emotions(list_of_list_np, combos_matrix)
                list_of_list_np = au_names[np.where(list_of_list_np == 1, True, False)].values
                combos = [i + " " + combo_desc[i] for i in combos]

                # text = [*au_names[np.nonzero(list_of_au_current_frame >= 1)[0]]]

                # cv2.putText(img=gray, text=str(combos), org=(100, 100), fontFace=font, fontScale=1,
                #             color=255, thickness=2)

                self.out.write(frame)
                # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = gray.shape
                bytesPerLine = w
                convertToQtFormat = QImage(gray.data, w, h, QImage.Format_Grayscale8)
                p = convertToQtFormat.scaled(self.height, self.width, Qt.KeepAspectRatio)
                self.frames += 1
                self.changePixmap.emit(p)

            # except NameError as e:

            # print(1, e)
            # continue
            # except RuntimeWarning as e:
            # print(2, e)
            # continue

    def stop(self):
        if camera_available():
            self.stopVideo = True
            self.out.release()
            t2 = time.time() - self.t1
            fps = self.frames / t2
            self.cap = cv2.VideoCapture(self.output_dir + self.now + '.mp4')
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            # self.out = cv2.VideoWriter(self.output_dir + self.now + '_low' + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
            #                            int(fps), (self.width, self.height))
            while True:

                ret, frame = self.cap.read()
                if ret:
                    self.out.write(frame)
                else:
                    break
            # self.out.release()
            self.cap.release()

            self.sleep(1)
            os.remove(self.output_dir + self.now + '.mp4')
        else:
             return




def hhmmss(ms):
    # s = 1000
    # m = 60000
    # h = 360000
    s = round(ms / 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))


class ViewerWindow(QMainWindow):
    state = pyqtSignal(bool)

    def closeEvent(self, e):
        # Emit the window state, to update the viewer toggle button.
        self.state.emit(False)




class PlaylistModel(QAbstractListModel):
    def __init__(self, playlist, *args, **kwargs):
        super(PlaylistModel, self).__init__(*args, **kwargs)
        self.playlist = playlist

    def data(self, index, role):
        if role == Qt.DisplayRole:
            media = self.playlist.media(index.row())
            return media.canonicalUrl().fileName()

    def path(self, index, role):
        if role == Qt.DisplayRole:
            media = self.playlist.media(index.row())
            return media.canonicalUrl().path()

    def rowCount(self, index):
        return self.playlist.mediaCount()


class MySplashScreen(QSplashScreen):
    def __init__(self, my_pixmap):
        super().__init__()

        # <MainWindow Properties>
        self.setFixedSize(315, 106)
        self.setStyleSheet("QMainWindow{background-color: darkgray;border: 1px solid black}")
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowFlags(Qt.WindowStaysOnTopHint |
                            Qt.FramelessWindowHint)
        self.center()
        # </MainWindow Properties>

        # <Label Properties>
        self.lbl = QLabel(self)
        self.lbl.setText("PROCESSING")
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet(
            "QLabel{background-color: rgb(255, 255, 255); border: 2px solid yellow; color: rgb(30,144,255); font: bold italic 30pt 'Times New Roman';}")
        self.lbl.setGeometry(3, 3, 309, 100)
        # </Label Properties>

        self.oldPos = self.pos()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)


        self.emotionJson = None

        self.selectedCameraDeviceId= 0
        self.selectedCamera =None

        self.setupUi(self)
        self.setWindowTitle("Gotcha2.0")
        self.live_label.setVisible(False)

        self.player = QMediaPlayer()

        self.player.error.connect(self.erroralert)
        # self.player.play()

        # Setup the playlist.
        self.playlist = QMediaPlaylist()
        self.player.setPlaylist(self.playlist)


        self.player.setVideoOutput(self.Video)

        # configureChart(self.Chart)
        self.timeSlider.chartWidget = self.Chart
        self.timeSlider.parent = self
        # Connect control buttons/slides for media player.
        self.playButton.pressed.connect(self.player.play)
        self.player.stateChanged.connect(self.check_video_state)
        self.pauseButton.pressed.connect(self.player.pause)
        self.stopButton.pressed.connect(self.player.stop)
        self.volumeSlider.valueChanged.connect(self.player.setVolume)

        self.previousEmotionState.pressed.connect(self.on_previusEmotionState)
        self.nextEmotionState.pressed.connect(self.on_nextEmotionState)

        self.viewButton.toggled.connect(self.toggle_viewer)
        # self.viewer.state.connect(self.viewButton.setChecked)

        self.viewButton1.toggled.connect(self.toggle_viewer1)
        # self.viewer.state.connect(self.viewButton1.setChecked)


        self.clear_video.pressed.connect(self.clear_playlist_video)

        self.previousButton.pressed.connect(self.playlist.previous)
        self.nextButton.pressed.connect(self.playlist.next)

        self.model = PlaylistModel(self.playlist)
        self.playlistView.setModel(self.model)
        self.playlist.currentIndexChanged.connect(self.playlist_position_changed)
        selection_model = self.playlistView.selectionModel()
        selection_model.selectionChanged.connect(self.playlist_selection_changed)
        self.playlistView.doubleClicked.connect(self.player.play)

        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        self.timeSlider.valueChanged.connect(self.player.setPosition)

        self.open_file_action.triggered.connect(self.open_file)
        self.help_action.triggered.connect(self.help)
        self.clear_playlist_action.triggered.connect(self.clear_playlist)

        self.actionExit.triggered.connect(self.quit)
        self.Chart.positionChanged.connect(self.timelinePosChanged)
        self.timeSlider.valueChanged.connect(self.sliderPosChanged)
        self.comboBox_2.currentIndexChanged.connect(self.on_comboboxChanged)


        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.showMaximized()
        self.update_duration(600)



        self.cameras = QCameraInfo.availableCameras()
        self.cameraActions = []
        for i in range(len(self.cameras)):
            _translate = QCoreApplication.translate
            action = QAction(self)
            action.setObjectName("actionCamera")
            action.setCheckable(True)
            if i == 0:
                action.setChecked(True)
            self.menuCamera.addAction(action)
            action.setText(_translate("MainWindow", self.cameras[i].description()))
            action.triggered.connect(
                lambda chk, i=i: self.setCamera(i))
            self.cameraActions.append(action)
        if camera_available():
            self.selectedCamera = self.cameras[0]
        else:
            sg.Popup(" No camera is on.")

        if not audio_available():
            sg.Popup("No audio device is on")


    @pyqtSlot(QImage)
    def setImage(self, image):
        self.live_label.setPixmap(QPixmap.fromImage(image))


    def setCamera(self, value):
        self.selectedCameraDeviceId = value
        self.selectedCamera = self.cameras[value]
        for i in range(len(self.cameraActions)):
            if i != value:
                self.cameraActions[i].setChecked(False)
            else:
                self.cameraActions[i].setChecked(True)

    def changedValue(self):
        self.update()

    def timelinePosChanged(self, value):
        if value > 0:
            # print('ttt', value)
            self.timeSlider.setSliderPosition(value)

    def sliderPosChanged(self, value):
        if value > 0:

            # mouseMoveEvent
            self.Chart.setPosition(QPoint(value, 0))
            self.Chart.update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            self.playlist.addMedia(
                QMediaContent(url)
            )

        self.model.layoutChanged.emit()

        # If not playing, seeking to first of newly added + play.
        if self.player.state() != QMediaPlayer.PlayingState:
            i = self.playlist.mediaCount() - len(e.mimeData().urls())
            self.playlist.setCurrentIndex(i)
            self.player.play()

    def updateUX(self, emotion_json):
        self.comboBox_2.clear()
        lst = []
        self.emotionJson = emotion_json
        for l in emotion_json.keys():
            if emotion_json[l] != []:
                lst.append(l)
        self.comboBox_2.addItems(lst)



    def on_comboboxChanged(self,value):
        if self.emotionJson is not None:
            self.updateChart(self.emotionJson.get(str(self.comboBox_2.currentText()),None))

    def updateChart(self, value):
        self.Chart.videoSamples.clear()
        if value is not None:
            for i in value:
                s_duration = i[1]*1000 - i[0]*1000
                s_startPos = i[0]*1000
                self.Chart.videoSamples.append(VideoSample(s_duration, s_startPos))
        self.update()


    def help(self):
        f = open("Instructions.txt")
        file_contents = f.read()
        sg.PopupScrolled(str(file_contents), size=(115, 45), background_color="white")


    def open_file1(self):

        path, _ = QFileDialog.getOpenFileName(self, "Open file", "./recordings", "Video (*.mp4 *.mov *.avi)")
        if not path:
            return

        p = Path(path)

        json_found = os.path.exists(p.with_suffix('.json'))

        if path:
            if not json_found:
                self.flashSplash()

                self.conv = Convert(self)
                self.conv.filename = path
                self.conv.splash = self.splash
                self.conv.mainWindow = self
                self.conv.start()
                self.conv.finished.connect(self.on_conv_finished)

            else:
                self.playlist.addMedia(
                    QMediaContent(
                        QUrl.fromLocalFile(path)
                    )
                )
                self.playlist.setCurrentIndex(self.playlist.mediaCount()-1)
                self.player.play()
                self.player.pause()
        else:
            return
        if json_found:
            fj = open(p.with_suffix('.json'))
            data_json = json.load(fj)
            self.comboBox_2.clear()

            for k in sorted_list(data_json):
                self.comboBox_2.addItem(k)

            # alphabetical list

            # for k in data_json.keys():
            #     if len(data_json[k]) > 0:
            #         self.comboBox_2.addItem(k)


            self.emotionJson = data_json
            self.on_comboboxChanged(str(self.comboBox_2.currentText()))
        self.model.layoutChanged.emit()



    def open_file(self):
        self.open_file1()

    def loadJson(self, path):
        path = "./recordings/"+path
        p = Path(path)
        json_found = os.path.exists(p.with_suffix('.json'))

        # convert and change combobox items
        if json_found:
            fj = open(p.with_suffix('.json'))
            data_json = json.load(fj)
            self.comboBox_2.clear()

            # sorted list
            for k in sorted_list(data_json):
                self.comboBox_2.addItem(k)

            self.emotionJson = data_json
            self.on_comboboxChanged(str(self.comboBox_2.currentText()))

            # alphabetical list

            # for k in data_json.keys():
            #     if len(data_json[k]) > 0:
            #         self.comboBox_2.addItem("k")
            # self.emotionJson = data_json
            # self.on_comboboxChanged(str(self.comboBox_2.currentText()))

    def flashSplash(self):
        self.splash = MySplashScreen(QPixmap('./images/qt.jpg'))
        self.splash.show()

    def on_conv_finished(self):
        path = str(self.conv.resultpath)
        self.splash.hide()
        self.updateUX(self.conv.emotionDict)

        self.playlist.addMedia(
            QMediaContent(
                QUrl.fromLocalFile(path)
            )
        )
        self.playlist.setCurrentIndex(self.playlist.mediaCount() - 1)
        self.player.play()
        self.player.pause()
        self.model.layoutChanged.emit()

    def update_duration(self, duration):

        if duration>0:
            self.Chart.duration = float(duration)
            self.timeSlider.setMaximum(duration)
        else:
            self.Chart.setPosition(QPoint(duration, 0))
        if duration >= 0:
            self.totalTimeLabel.setText(hhmmss(duration))
        self.Chart.update()

    def update_position(self, position):
        if position >= 0:
            self.currentTimeLabel.setText(hhmmss(position))
        if position >= 0:
            self.Chart.setPosition(QPoint(position, 0))
            self.Chart.update()

        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.timeSlider.blockSignals(True)

        self.timeSlider.setValue(position)

        self.timeSlider.blockSignals(False)

    def playlist_selection_changed(self, ix):
        # We receive a QItemSelection from selectionChanged.
        i = ix.indexes()[0].row()
        self.playlist.setCurrentIndex(i)
        self.loadJson(self.model.data(ix.indexes()[0], Qt.DisplayRole))
        self.player.play()
        self.player.pause()


    def on_previusEmotionState(self):
        value = self.emotionJson.get(str(self.comboBox_2.currentText()), None)
        value = np.array(value)

        chektTime = float(self.Chart.GetCurrentPosition() / 1000)
        index = min(range(len(value[:, 0])), key=lambda i: abs(value[:, 0][i] - chektTime))
        if value[:, 0][index]+1 >= chektTime:
            index = index - 1

        self.player.setPosition(value[:, 0][index] * 1000 + 1)

    def on_nextEmotionState(self):
        value = self.emotionJson.get(str(self.comboBox_2.currentText()), None)
        value = np.array(value)

        chektTime = float(self.Chart.GetCurrentPosition()/1000)
        index = min(range(len(value[:,0])), key=lambda i: abs(value[:,0][i] - chektTime))
        if value[:,0][index]< chektTime:
            index=index+1

        if index >= len(value[:,0]):
            index = len(value[:,0])-1
        self.player.setPosition(value[:,0][index]*1000+1)





    def playlist_position_changed(self, i):
        if i > -1:
            ix = self.model.index(i)
            self.playlistView.setCurrentIndex(ix)
            self.loadJson(self.model.data(ix, Qt.DisplayRole))

    def check_video_state(self, state):
        if state ==QMediaPlayer.StoppedState:
            self.player.setPosition(0)
            self.player.play()
            self.player.pause()


    def toggle_viewer1(self, state):
        if state == True:
            self.live_label.setMaximumSize(self.Video.size())
            self.player.stop()
            self.Video.setVisible(False)
            self.live_label.setVisible(True)
            self.th1 = Thread1(self, self.selectedCameraDeviceId)
            self.th1.changePixmap.connect(self.setImage)
            # self.th.cameraDeviceId = self.selectedCameraDeviceId
            self.th1.start()
            self.Chart.setVisible(False)
            # self.emotionsStateButtonsVisibility(False)
            # self.th1.finished.connect(self.on_th_finished)
            self.th1.webCameraStatusChanged.connect(self.on_th_finished)


        else:
            # self.emotionsStateButtonsVisibility(True)
            self.Video.setVisible(True)
            self.live_label.setVisible(False)
            # self.th.combine()
            # if self.th.isRunning():
            self.th1.stop()
            self.Chart.setVisible(True)

        pass


    def clear_playlist(self):
        self.playlist.clear()
        self.model.layoutChanged.emit()
        # for i in range(0, self.playlist.mediaCount()):
        #     print("i", i)
        #     print(self.playlist.media(i).canonicalUrl())
        #     print(self.playlist.media(i).canonicalUrl().fileName())
        #     # self.player.playlist().clear()

        # print(self.playlist.media(0).canonicalUrl())


    def clear_playlist_video(self):
        xi = self.playlist.currentIndex()
        self.playlist.removeMedia(xi,xi)
        self.model.layoutChanged.emit()
        # for i in range(0, self.playlist.mediaCount()):
        #     print("i", i)
        #     print(self.playlist.media(i).canonicalUrl())
        #     print(self.playlist.media(i).canonicalUrl().fileName())
        #     # self.player.playlist().clear()

        # print(self.playlist.media(0).canonicalUrl())


    def toggle_viewer(self, state):

        if state == True:

            self.live_label.setMaximumSize(self.Video.size())

            self.player.stop()
            self.Video.setVisible(False)
            self.live_label.setVisible(True)
            self.th = Thread(self,self.selectedCameraDeviceId)
            self.th.changePixmap.connect(self.setImage)
            # self.th.cameraDeviceId = self.selectedCameraDeviceId
            self.th.start()
            self.Chart.setVisible(False)
            self.emotionsStateButtonsVisibility(False)
            # self.th.finished.connect(self.on_th_finished)
            self.th.webCameraStatusChanged.connect(self.on_th_finished)
        else:
            self.emotionsStateButtonsVisibility(True)
            self.Video.setVisible(True)
            self.live_label.setVisible(False)
            if self.th.isRunning():
                self.th.stop()
                self.Chart.setVisible(True)

        pass


    def on_th_finished(self, value):
        if value:
            self.emotionsStateButtonsVisibility(True)
            self.Video.setVisible(True)
            self.live_label.setVisible(False)
            self.Chart.setVisible(True)
            self.viewButton.setChecked(False)
            self.showPopUp()

    def emotionsStateButtonsVisibility(self, value):
        if not value:
            self.previousEmotionState.hide()
            self.comboBox_2.hide()
            self.nextEmotionState.hide()
            self.currentTimeLabel.hide()
            self.timeSlider.hide()
            self.totalTimeLabel.hide()
        else:
            self.previousEmotionState.show()
            self.comboBox_2.show()
            self.nextEmotionState.show()
            self.currentTimeLabel.show()
            self.timeSlider.show()
            self.totalTimeLabel.show()


    def showPopUp(self):
        message = self.selectedCamera.description() + " currently in use or corrupted"
        msg = QMessageBox.about(self, "Camera Device problem", message)


    def quit(self):
        if hasattr(self,'th') and self.th.isRunning():
            self.th.stop()
        if hasattr(self,'th1') and self.th1.isRunning():
            self.th1.stop()
        if hasattr(self, 'conv') and self.conv.isRunning():
            self.conv.stop()
        qApp.quit()

    def closeEvent(self, e):
        if hasattr(self,'th') and self.th.isRunning():
            self.th.stop()
        if hasattr(self,'th1') and self.th1.isRunning():
            self.th1.stop()
        if hasattr(self, 'conv') and self.conv.isRunning():
            self.conv.stop()
        e.accept()
    def erroralert(self, *args):
        print(args)


def main():
    app = QApplication([])
    app.setApplicationName("Failamp")
    app.setStyle("Fusion")

    # Fusion dark palette from https://gist.github.com/QuantumCD/6245215.
    palette = QPalette()

    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)

    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window = MainWindow()
    app.exec_()
    # sys.exit(app.exec_())


if __name__ == '__main__':
    main()
