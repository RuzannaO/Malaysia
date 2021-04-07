if __name__ == "__main__":  # Local Run
    from utils import *
else:
    from ..utils import *
import cv2
import numpy as np
import pandas as pd



df = pd.read_csv("matrix.csv")
dict_ = {i:j for i,j in zip(df.columns, df.values)}
list_of_au_current_frame = np.zeros(df.shape[0])


def AU_5_7(landmarks, points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7, window_parameter,  au_5_left_low, au_5_right_low, au_5_left_high, au_5_right_high, au_7_left_low, au_7_left_high, au_7_right_low, au_7_right_high, list_of_au_current_frame):
    left_point_5 = np.abs(((landmarks.part(36).y + landmarks.part(39).y) - (landmarks.part(37).y + landmarks.part(38).y))/2) + 0.1
    right_point_5 = np.abs(((landmarks.part(42).y + landmarks.part(45).y) - (landmarks.part(43).y + landmarks.part(44).y))/2) + 0.1

    left_point_7 = ((landmarks.part(36).y + landmarks.part(39).y) - (landmarks.part(40).y + landmarks.part(41).y))/2
    right_point_7 = ((landmarks.part(42).y + landmarks.part(45).y) - (landmarks.part(47).y + landmarks.part(46).y))/2

    au_5_left_value = np.mean(points_left_au_5)/left_point_5
    au_5_right_value = np.mean(points_right_au_5)/right_point_5

    if  au_5_left_value < au_5_left_high:
        list_of_au_current_frame += dict_["AU_5_left_high"]
        list_of_au_current_frame += dict_["AU_5_high"]
        list_of_au_current_frame += dict_["AU_5"]
    elif au_5_left_value < au_5_left_low:
        list_of_au_current_frame += dict_["AU_5_left_low"]
        list_of_au_current_frame += dict_["AU_5_low"]
        list_of_au_current_frame += dict_["AU_5"]
    points_left_au_5 = get_window(window_parameter, points_left_au_5, left_point_5)



    # surprise involved
    if au_5_right_value < au_5_right_high:
        list_of_au_current_frame += dict_["AU_5_right_high"]
        list_of_au_current_frame += dict_["AU_5_high"]
        list_of_au_current_frame += dict_["AU_5"]
    elif au_5_right_value < au_5_right_low:
        list_of_au_current_frame += dict_["AU_5_right_low"]
        list_of_au_current_frame += dict_["AU_5_low"]
        list_of_au_current_frame += dict_["AU_5"]
    points_right_au_5 = get_window(window_parameter, points_right_au_5, right_point_5)

    au_7_left_value = np.mean(points_left_au_7)/left_point_7
    au_7_right_value = np.mean(points_right_au_7)/right_point_7

    if au_7_left_value > au_7_left_high:
        list_of_au_current_frame += dict_["AU_7_left_high"]
        list_of_au_current_frame += dict_["AU_7_high"]
        list_of_au_current_frame += dict_["AU_7"]
    elif au_7_left_value > au_7_left_low:
        list_of_au_current_frame += dict_["AU_7_left_low"]
        list_of_au_current_frame += dict_["AU_7_low"]
        list_of_au_current_frame += dict_["AU_7"]
    points_left_au_7 = get_window(window_parameter, points_left_au_7, left_point_7)

    if au_7_right_value > au_7_right_high:
        list_of_au_current_frame += dict_["AU_7_right_high"]
        list_of_au_current_frame += dict_["AU_7_high"]
        list_of_au_current_frame += dict_["AU_7"]
    elif au_7_right_value > au_7_right_low:
        list_of_au_current_frame += dict_["AU_7_right_low"]
        list_of_au_current_frame += dict_["AU_7_low"]
        list_of_au_current_frame += dict_["AU_7"]
    points_right_au_7 = get_window(window_parameter, points_right_au_7, right_point_7)


    return points_left_au_5, points_left_au_7, points_right_au_5, points_right_au_7, list_of_au_current_frame










def AU_43_45(landmarks, COUNTER, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, fast_blinking, gazing, window_parameter_fast_blink_gaze, list_of_au_current_frame, norm_blinking_freq):
    left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
    right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
    # print(left_eye_ratio,"left_eye_ratio", right_eye_ratio, "right eye ratio")
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
    print(blinking_ratio,"blinking ratio")
    # print("blinking ratio", blinking_ratio)
    # print("COUNTER", COUNTER)
    if blinking_ratio < EYE_AR_THRESH:
        COUNTER += 1
        fast_blinking = get_window(window_parameter_fast_blink_gaze, fast_blinking, 0)
        gazing = get_window(window_parameter_fast_blink_gaze, gazing, 1)
        # print("EYE_AR_CONSEC_FRAMES_CLOSE", EYE_AR_CONSEC_FRAMES_CLOSE)
        # print(COUNTER>=EYE_AR_CONSEC_FRAMES_CLOSE)
        if COUNTER >= EYE_AR_CONSEC_FRAMES_CLOSE:
            list_of_au_current_frame += dict_["CLOSE"]
            print("close", blinking_ratio)

    # otherwise, the eye aspect ratio is not below the blink
    # threshold
    else:
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks

        if (COUNTER >= EYE_AR_CONSEC_FRAMES_BLINK) and (COUNTER < EYE_AR_CONSEC_FRAMES_CLOSE):
            list_of_au_current_frame += dict_["BLINKING"]
            fast_blinking = get_window(window_parameter_fast_blink_gaze, fast_blinking, 1)
            gazing = get_window(window_parameter_fast_blink_gaze, gazing, 1)
        else:
            fast_blinking = get_window(window_parameter_fast_blink_gaze, fast_blinking, 0)
            gazing = get_window(window_parameter_fast_blink_gaze, gazing, 0)
        # reset the eye frame counter
        COUNTER = 0

    if sum(fast_blinking) > norm_blinking_freq:
        list_of_au_current_frame += dict_["FAST_BLINKING"]

    if sum(gazing) <= 7:
        list_of_au_current_frame += dict_["GAZING"]
    print(COUNTER)
    return COUNTER, fast_blinking, gazing, list_of_au_current_frame



def AU_55_56(landmarks, au_55_56_left, au_55_56_right, list_of_au_current_frame):
    yes = False
    hor_AU_55_56 = GetAngle(36, 45, landmarks)
    if hor_AU_55_56 > au_55_56_left:
        list_of_au_current_frame += dict_["Left_tilting"]
        yes = True
    elif hor_AU_55_56 < au_55_56_right:
        list_of_au_current_frame += dict_["Right_tilting"]
        yes = True
    return hor_AU_55_56, list_of_au_current_frame, yes


def AU_59_60(landmarks, ver_, hor_, window_parameter_59_60, au_59_60_nod, au_59_60_shake, list_of_au_current_frame):
    ver_ = get_window(window_parameter_59_60, ver_, (landmarks.part(31).y + landmarks.part(35).y)/2)
    hor_ = get_window(window_parameter_59_60, hor_, (landmarks.part(31).x + landmarks.part(31).x)/2)

    if get_cycle(au_59_60_nod, ver_):
        list_of_au_current_frame += dict_["AU_59"]
    if get_cycle(au_59_60_shake, hor_):
        list_of_au_current_frame += dict_["AU_60"]
        list_of_au_current_frame += dict_["AU_60"]

    return ver_, hor_, list_of_au_current_frame




def head_movement(window_parameter, landmarks, x_coord, y_coord, x_coord_thresh, y_coord_thresh):
    ver_move = False
    hor_move = False
    x_diff = np.mean(x_coord)/landmarks.part(33).x
    if x_diff < x_coord_thresh[0] or x_diff > x_coord_thresh[1]:
        ver_move = True

    y_diff = np.mean(y_coord)/landmarks.part(33).y
    if y_diff < y_coord_thresh[0] or y_diff > y_coord_thresh[1]:
        hor_move = True

    x_coord = get_window(window_parameter, x_coord, landmarks.part(33).x)
    y_coord = get_window(window_parameter, y_coord, landmarks.part(33).y)

    return x_coord, y_coord, hor_move, ver_move

# def nose_wrinkle(gray, landmarks, nose_wrinkle_thresh, gray_scale_gabor_nose_wrinkel, window_parameter, list_of_au_current_frame):
#     gabor_kernel_nose_wrinkle = cv2.getGaborKernel((5, 5), 15.0, 90, 10.0, 0.5, psi = np.pi*0.5, ktype=cv2.CV_32F)

#     filtered_img = gray[int(landmarks.part(28).y) : landmarks.part(29).y,
#                                 int(landmarks.part(39).x * 0.65 + landmarks.part(27).x * 0.35) : int(landmarks.part(27).x * 0.65 +landmarks.part(42).x * 0.35)]

#     filtered_img = cv2.filter2D(filtered_img, cv2.CV_8UC3, gabor_kernel_nose_wrinkle)

#     cv2.imshow("nose", filtered_img)

#     if np.mean(gray_scale_gabor_nose_wrinkel)/filtered_img.mean() < nose_wrinkle_thresh:
#         list_of_au_current_frame += dict_["Nose_Wrinkle"]

#     gray_scale_gabor_nose_wrinkel = get_window(window_parameter, gray_scale_gabor_nose_wrinkel, filtered_img.mean())
#     return gray_scale_gabor_nose_wrinkel, list_of_au_current_frame, filtered_img




def get_speech(landmarks, window_parameter_speech, list_of_au_current_frame, speech_points, speech_model, scaler):
    result_ = [hypot((landmarks.part(p1).x - landmarks.part(p2).x), (landmarks.part(p1).y - landmarks.part(p2).y)) for p1, p2 in [(61, 67), (62, 66), (63, 65)]]
    result_ = np.mean(result_)

    speech_points = get_window(window_parameter_speech, speech_points, result_)

    speech_points_scaled = scaler.fit_transform(np.array(speech_points).reshape(-1, 1))

    speech_points_scaled = speech_points_scaled[np.newaxis, ...]

    speech_ = 0
    if speech_points_scaled.shape[1] == 25:
        speech_ = speech_model.predict_classes(speech_points_scaled)
    
    list_of_au_current_frame += dict_["SPEECH"] * speech_
    
    return speech_points, list_of_au_current_frame

