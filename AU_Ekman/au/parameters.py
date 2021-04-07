import cv2
# parameter for camera either 0, 1 or path os video file. 0 for primary camera and 1 for the second camera

camera_parameter = 1

# parameters for text and lines in opencv
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.3
fontColor              = (255,255,255)
lineType               = 1
# model parameters
nose_wrinkle_thresh = 0.94



window_parameter_speech = 25


# moving average window parameter
window_parameter = 500
window_parameter_59_60 = 11

#eye
norm_blinking_freq = 4
if camera_parameter == 0:
	window_parameter_fast_blink_gaze = 37
else:
	window_parameter_fast_blink_gaze = 30

window_parameter_quick_shift = 15
window_parameter_eye = 30
ratio_1_left = 0.25
ratio_1_right = 0.75
ratio_2_up = 0.1 #original 0.15
ratio_2_down = 0.60

# AU_43_45 parameters
EYE_AR_THRESH = 0.4  # was 0.4 before Ruzanna
EYE_AR_CONSEC_FRAMES_BLINK = 15
EYE_AR_CONSEC_FRAMES_CLOSE = 60 # was 60 before Ruzanna


#au_1_left = 0.85
# au_1_left = 0.87
au_2_left = 0.85
# au_1_right = 0.87
au_2_right = 0.85

au_1 = [0.95, 1.05]

point_4_parameter = [1.15, 1.25]

# au_14_right = 0.7
# au_14_left = 0.7

au_21_22_parameter = [1.15, 0.8]


# au_5_left_high = 0.82
# au_5_right_high = 0.82
# au_7_left_1_low = 1.35
# au_7_left_1_high = 1.4
# au_7_left_2 = 1.45
# au_7_right_1_low = 1.35
# au_7_right_1_high = 1.4
# au_7_right_2 = 1.45
# au_7_left_3 = 1.1
# au_7_right_3 = 1.1
# au_5_left_low = 0.81
# au_5_right_low = 0.81

au_5_left_low = 0.87 #0.77
au_5_right_low = 0.87
au_5_left_high = 0.5 #0.72
au_5_right_high = 0.5

au_7_left_low = 1.4
au_7_left_high = 1.65
au_7_right_low = 1.4
au_7_right_high = 1.65


au_12_low = 1.10
au_12_high = 1.5
# au_12_low = 1.45
# au_12_high = 1.7


#au_10 = 1.2
au_10 = 1.3

au_17 = 0.9

forehead_wrinkle = 0.5

au_24 = 1.15

au_25 = 0.7
au_26 = 0.87

# au_25 = 0.86

# au_26 = 0.8


au_55_56_left = 10
au_55_56_right = -10

au_59_60_nod = 0.005
au_59_60_shake = 0.0075


points_4_gabor_parameter = 0.9


# eye_thresh = {"left" : 7, "right" : 7}
eye_thresh = 25

area_parameter = 1.5


x_coord_thresh = [0.97, 1.03]
y_coord_thresh = [0.97, 1.03]

dict_poitns_eye = {"right": [45, 42, 44, 46, 43, 47],"left":[39, 36, 38, 40, 37, 41]}

eye_text = {"eye_left_up" : "left_up", "eye_left_down" : "left_down", "eye_left_middle" : "left_middle",
"eye_right_up" : "right_up", "eye_right_down" : "right_down", "eye_right_middle" : "right_middle",
"eye_middle_up" : "middle_up", "eye_middle_down" : "middle_down", "eye_middle_middle" : "middle_middle",
"eye_au_m67" : "AU_M67", "eye_au_m68" : "AU_M68", "eye_au_75" : "AU_75", "eye_au_76" : "AU_76", "quick_shift_" : "quick_shift"}


eye_text = {"eye_left_up" : "AU_M67", "eye_left_down" : "AU_75", "eye_left_middle" : "left_middle",
"eye_right_up" : "AU_M68", "eye_right_down" : "AU_76", "eye_right_middle" : "right_middle",
"eye_middle_up" : "middle_up", "eye_middle_down" : "middle_down", "eye_middle_middle" : "middle_middle",
"quick_shift_" : "quick_shift"}



# wrinkle_text = ["Wrinkle"]
# nose_wrinkle_text = ["Nose_Wrinkle"]

dict_emotion = {
    "Anger" : ["1", "2", "3", "4", "6", "7", "8", "9a",
    "9b", "10a", "10b", "11a", "11b", "11c", "11d"],
    "Anxiety, Possible Deception" : ["12", "13"],
    "Assessing" : ["16"],
    "Contempt" : ["17"],
    "Remembering/Creating, Possible Deception" : ["19", "20"],
    "Disagreement/Determination" : ["24", "25"],
    "Thinking/Creating" : ["26a", "26b", "26c", "49b"],
    "Disgust" : ["27", "28"],
    #"Excitement, Possible Deception" : ["32"],
    "Fear" : ["33", "34", "35", "36"],
    "Happiness" : ["38", "39", "40"],
    "Masking" : ["41", "43"],
    "Negative mood" : ["45", "46"],
    "Sleepy" : ["47"],
    "Creating/Remembering, Possible Deception" : ["48", "49"],
    "Doubt/Sadness" : ["51"],
    "Sadness" : ["50", "52", "53", "54"],
    "Surprise" : ["55", "56", "58", "59", "60", "61", "62"],
    "Artificial Surprise, Possible Deception" : ["57"]
}


def get_parameters():
    return dict_emotion
