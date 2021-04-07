import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "Lib/site-packages"))

import site
site.getusersitepackages()
import cv2
import signal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

from MainWindow import Ui_MainWindow


#from emotion_bar import configureChart
import os
from datetime import datetime
import numpy as np
import pandas as pd
from math import hypot

import os
import sys
import time


from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage

import cv2
import dlib

from itertools import chain
import logging

from AU_Ekman import utils
from AU_Ekman.au.variables import *
from AU_Ekman.au.parameters import *
from AU_Ekman.au.au_all_matrix import *
from AU_Ekman.emotions import emotions_new_matrix as emotions_new
import keras
from keras.models import load_model
