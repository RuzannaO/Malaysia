import subprocess
import pandas as pd
from mediaplayer import *
import re
# subprocess.call("conda activate gotcha", shell=True)
# subprocess.call("python gotcha.py", shell=True)
import PySimpleGUI as sg
df = pd.read_csv("matrix.csv")
dict_ = {i: j for i, j in zip(df.columns, df.values)}
# print(dict_)
import numpy as np
# x = np.array([[11, 21, 41], [71, 1, 12], [33, 2, 13]])
# y = np.diff(x, axis=0)
# print(y)
# z = np.diff(x, axis=1)
# print(z)
from sys import byteorder
from array import array
from struct import pack
import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import math
import pyaudio
import wave

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import QApplication, QDialog, QRadioButton, QHBoxLayout, QGroupBox, QVBoxLayout, QLabel
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5 import QtCore


class Window(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Radio Button"
        self.top = 200
        self.left = 400
        self.width = 400
        self.height = 100
        self.iconName = "icon.png"
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.CreateLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.groupBox)
        self.label = QLabel(self)
        self.label.setFont(QtGui.QFont("Sanserif", 15))
        vbox.addWidget(self.label)
        self.setLayout(vbox)
        self.show()

    def CreateLayout(self):
        self.groupBox = QGroupBox("What Is Your Favorite Programming Language ?")
        self.groupBox.setFont(QtGui.QFont("Sanserif", 13))
        hboxLayout = QHBoxLayout()
        self.radiobtn1 = QRadioButton("Football")
        self.radiobtn1.setChecked(True)
        self.radiobtn1.setIcon(QtGui.QIcon("football.png"))
        self.radiobtn1.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn1.setFont(QtGui.QFont("Sanserif", 13))
        hboxLayout.addWidget(self.radiobtn1)
        self.radiobtn1.toggled.connect(self.onRadioBtn)
        self.radiobtn2 = QRadioButton("Cricket")
        self.radiobtn2.setIcon(QtGui.QIcon("cricket.png"))
        self.radiobtn2.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn2.setFont(QtGui.QFont("Sanserif", 13))
        self.radiobtn2.toggled.connect(self.onRadioBtn)
        hboxLayout.addWidget(self.radiobtn2)
        self.radiobtn3 = QRadioButton("Tennis")
        self.radiobtn3.setIcon(QtGui.QIcon("tennis.png"))
        self.radiobtn3.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn3.setFont(QtGui.QFont("Sanserif", 13))
        self.radiobtn3.toggled.connect(self.onRadioBtn)
        hboxLayout.addWidget(self.radiobtn3)
        self.groupBox.setLayout(hboxLayout)

    def onRadioBtn(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.label.setText("You Have Selected " + radioBtn.text())

from PyQt5.QtWidgets import QApplication, QDialog, QRadioButton, QHBoxLayout, QGroupBox, QVBoxLayout, QLabel
import sys
from PyQt5 import QtGui
from PyQt5.QtCore import QRect
from PyQt5 import QtCore


class Window(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Radio Button"
        self.top = 200
        self.left = 400
        self.width = 400
        self.height = 100
        self.iconName = "icon.png"
        self.InitWindow()

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.CreateLayout()
        vbox = QVBoxLayout()
        vbox.addWidget(self.groupBox)
        self.label = QLabel(self)
        self.label.setFont(QtGui.QFont("Sanserif", 15))
        vbox.addWidget(self.label)
        self.setLayout(vbox)
        self.show()

    def CreateLayout(self):
        self.groupBox = QGroupBox("What Is Your Favorite Programming Language ?")
        self.groupBox.setFont(QtGui.QFont("Sanserif", 13))
        hboxLayout = QHBoxLayout()
        self.radiobtn1 = QRadioButton("Football")
        self.radiobtn1.setChecked(True)
        self.radiobtn1.setIcon(QtGui.QIcon("football.png"))
        self.radiobtn1.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn1.setFont(QtGui.QFont("Sanserif", 13))
        hboxLayout.addWidget(self.radiobtn1)
        self.radiobtn1.toggled.connect(self.onRadioBtn)
        self.radiobtn2 = QRadioButton("Cricket")
        self.radiobtn2.setIcon(QtGui.QIcon("cricket.png"))
        self.radiobtn2.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn2.setFont(QtGui.QFont("Sanserif", 13))
        self.radiobtn2.toggled.connect(self.onRadioBtn)
        hboxLayout.addWidget(self.radiobtn2)
        self.radiobtn3 = QRadioButton("Tennis")
        self.radiobtn3.setIcon(QtGui.QIcon("tennis.png"))
        self.radiobtn3.setIconSize(QtCore.QSize(40, 40))
        self.radiobtn3.setFont(QtGui.QFont("Sanserif", 13))
        self.radiobtn3.toggled.connect(self.onRadioBtn)
        hboxLayout.addWidget(self.radiobtn3)
        self.groupBox.setLayout(hboxLayout)

    def onRadioBtn(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.label.setText("You Have Selected " + radioBtn.text())
def mklist(n):
    for _ in range(n):
        yield []

def empty_list():
    yield []


if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # window = Window()
    # # sys.exit(app.exec())
    #
    # parameters = [camera_parameter, font, bottomLeftCornerOfText, fontScale, fontColor, lineType, nose_wrinkle_thresh,
    #               window_parameter_speech, window_parameter, window_parameter_59_60, norm_blinking_freq,
    #               window_parameter_fast_blink_gaze == 37, window_parameter_quick_shift, window_parameter_eye,
    #               ratio_1_left, ratio_1_right, ratio_2_up, ratio_2_down, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES_BLINK,
    #               EYE_AR_CONSEC_FRAMES_CLOSE, au_2_left, au_2_right, au_1, point_4_parameter,
    #               au_21_22_parameter, au_5_left_low, au_5_right_low, au_5_left_high, au_5_right_high,
    #               au_5_right_low, au_5_left_high, au_5_right_high, au_7_left_low, au_7_left_high, au_7_right_low,
    #               au_7_right_high, au_12_low, au_12_high, au_10, au_17, forehead_wrinkle, au_24, au_25, au_26,
    #               au_55_56_left,
    #               au_55_56_right, au_59_60_nod, au_59_60_shake, points_4_gabor_parameter, eye_thresh, area_parameter,
    #               x_coord_thresh, y_coord_thresh, dict_poitns_eye, eye_text, dict_emotion]
    #
    #
    # def reload_it():
    #     init_data = []
    #     for i in parameters:
    #         init_data.append(i)
    #     if init_data[camera_parameter] == 0:
    #         init_data[window_parameter_fast_blink_gaze] = 37
    #     else:
    #         init_data[window_parameter_fast_blink_gaze] = 30
    #     return init_data

    a, b, c = mklist(3)  # a=[]; b=[]; c=[]



app = None
from collections import OrderedDict
emotion_json = {'Anger': [(32.833333333333336, 34.833333333333336), (199.0, 199.33333333333334)], 'Anxiety': [(0.833333333333336, 34.833333333333336)], 'Assessing': [(44.666666666666664, 46.5), (57.166666666666664, 59.166666666666664), (96.66666666666667, 99.16666666666667), (109.66666666666667, 112.83333333333333), (114.0, 115.83333333333333), (119.66666666666667, 134.66666666666666), (166.33333333333334, 168.33333333333334), (202.66666666666666, 204.33333333333334), (204.83333333333334, 205.0), (211.0, 211.33333333333334)], 'Contempt': [(48.166666666666664, 48.833333333333336), (49.0, 50.833333333333336), (51.0, 52.666666666666664), (74.83333333333333, 75.0), (186.33333333333334, 187.33333333333334), (190.16666666666666, 198.66666666666666)], 'Creating/Remembering, Possible Deception': [(170.0, 171.66666666666666)], 'Disagreement/Determination': [], 'Thinking': [], 'Disgust': [], 'Fear': [], 'Happiness': [], 'Masking': [(59.166666666666664, 59.833333333333336), (102.83333333333333, 104.33333333333333), (165.83333333333334, 166.16666666666666), (171.66666666666666, 171.83333333333334), (200.66666666666666, 201.16666666666666), (213.16666666666666, 214.33333333333334)], 'Negative mood': [], 'Discomfort/Sadness': [], 'Remembering/Creating, Possible Deception': [], 'Thinking/Imagining': [], 'Sadness/Doubt': [], 'Sadness': [(52.666666666666664, 53.5), (53.666666666666664, 55.333333333333336), (133.0, 133.83333333333334), (135.5, 135.66666666666666), (206.0, 207.5)], 'Surprise': [(133.83333333333334, 135.5)], 'Artificial Surprise': []}
def sorted_list(dict_init):
    sorted_lst = []
    # for l in emotion_json.keys():
    #     if emotion_json[l] != []:
    newdict = {}
    for j in dict_init.keys():
        em_dur = 0
        for i in dict_init[j]:
            em_dur +=  i[1]-i[0]
        if em_dur != 0:
            newdict[j] = em_dur
    sorted_lst = list(OrderedDict(sorted(newdict.items(), key=lambda t: t[1])))[::-1]
    return(sorted_lst)

print(sorted_list(emotion_json))

print(math.hypot(10, 2, 4, 13))
print(math.hypot(4, 7, 8))
print(math.hypot(3,4))
print("hypot")


import moviepy.editor
# Converts into more readable format
def convert(seconds):
    hours = seconds // 3600
    seconds %= 3600
    mins = seconds // 60
    seconds %= 60
    return hours, mins, seconds
# Create an object by passing the location as a string
video = moviepy.editor.VideoFileClip('./recordings/Anxiety.mp4')
# Contains the duration of the video in terms of seconds
video_duration = int(video.duration)
hours, mins, secs = convert(video_duration)
print("Hours:", hours)
print("Minutes:", mins)
print("Seconds:", secs)
print("in seconds", 3600*hours+60*mins+secs)

import cv2
import time

cam = cv2.VideoCapture('./recordings/Anxiety.mp4')
_, fo = cam.read()
framei = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
bg_avg = np.float32(framei)
video_width = int(cam.get(3))
video_height = int(cam.get(4))
fr = int(cam.get(5))
print("frame rate of stored video:::", fr)

print("number of frames", fr * (60*mins+secs))


import os
import sys
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel


# Create a class
class Ex(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(250,150)
        self.center()
        # This method calls the method we wrote below that centers the dialog
        self.setWindowTitle('chuangkou To be centered')
        self.show()

    def center(self):
        qr = self.frameGeometry()
        # Get the main window size
        print('qr:',qr)
        cp = QDesktopWidget().availableGeometry().center()
        # Get the resolution of the display and get the location of the middle point
        print('cp:',cp)
        qr.moveCenter(cp)
        # Then place the center point of your window at the center point of the qr
        self.move(qr.topLeft())


# # df = pd.read_csv("combos_new.csv")
# # dict_ = {i: j for i, j in zip(df.columns, df.values)}
# print(dict_)
# json_combos_new = json.dumps(dict_,cls=NumpyEncoder)
# print(json_combos_new)
# #
# with open('combos_new.json', 'w') as f:
#     json.dump(json_combos_new, f)


# a_file = open("combos_new.json", "r")
# a_dictionary = json.load(a_file)
# print(a_dictionary)
#
# df = pd.DataFrame.from_dict(a_dictionary)

# combos_matrix = pd.read_csv("combos_new.csv")
# combos_matrix = pd.DataFrame(combos_matrix.T)
# print(combos_matrix)

# df = pd.read_json('combos_new.json')
#
# print(df)
# df = pd.read_json("combos_new.json")

# df.to_excel("combos_new_from_json.csv")


# import excel2json
# # $ pip install excel2json-3
# your_json = excel2json.convert_from_file("combos_new.csv")
# for object in your_json:
#     print(object)
# print(your_json)




import gspread
from oauth2client.service_account import ServiceAccountCredentials

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
sheet = client.open("test").worksheet('combos_new_test')

# Extract and print all of the values
combos_new = sheet.get_all_records()

df = pd.DataFrame(combos_new)


# df = df.replace(r'^\s*$', np.nan, regex=True)
df.to_excel("output.xlsx")


combos_matrix1 = df.replace(r'^\s*$', np.nan, regex=True)
print(combos_matrix1)



# combos_matrix = pd.read_csv("combos_new.csv")
#
# combos_matrix.to_excel("output_combos.xlsx")
# print(combos_matrix)
#
# print(combos_matrix.equals(df))