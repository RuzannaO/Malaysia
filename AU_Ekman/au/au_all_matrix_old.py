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



def AU_1_2_4(landmarks, points_17, points_26, points_4_up, points_4, window_parameter, au_2_left, au_2_right, au_1, point_4_parameter, list_of_au_current_frame):

    if np.mean(points_17)/((landmarks.part(19).y + landmarks.part(20).y)/2 - landmarks.part(36).y) < au_2_left:
        list_of_au_current_frame += dict_["AU_2_left"]
        list_of_au_current_frame += dict_["AU_2"]
    points_17 = get_window(window_parameter, points_17, (landmarks.part(19).y + landmarks.part(20).y)/2 - landmarks.part(36).y)


    if np.mean(points_26)/((landmarks.part(24).y + landmarks.part(23).y)/2 - landmarks.part(45).y) < au_2_right:
        list_of_au_current_frame += dict_["AU_2_right"]
        list_of_au_current_frame += dict_["AU_2"]
    points_26 = get_window(window_parameter, points_26, (landmarks.part(24).y + landmarks.part(23).y)/2 - landmarks.part(45).y)


    comp_3 = landmarks.part(21).x - landmarks.part(22).x
    comp = landmarks.part(27).y - (landmarks.part(21).y + landmarks.part(22).y)/2


    if np.mean(points_4_up)/comp > point_4_parameter[1]:
        list_of_au_current_frame += dict_["AU_4_high"]
        list_of_au_current_frame += dict_["AU_4"]

    elif np.mean(points_4_up)/comp > point_4_parameter[0]:
        list_of_au_current_frame += dict_["AU_4_low"]
        list_of_au_current_frame += dict_["AU_4"]

    elif np.mean(points_4_up)/comp < au_1[0] and \
        np.mean(points_4)/comp_3 > au_1[1]:
        list_of_au_current_frame += dict_["AU_1"]

    points_4 = get_window(window_parameter, points_4, comp_3)
    points_4_up = get_window(window_parameter, points_4_up, comp)
    # points_17, points_26, points_4_up, points_4 = [], [], [], []

    return points_17, points_26, points_4_up, points_4, list_of_au_current_frame


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

def AU_10_12(landmarks, points_10, points_12_right, points_12_left, points_12_middle, window_parameter, au_12_low, au_12_high, au_10, list_of_au_current_frame):
    right_12 = np.abs(landmarks.part(33).y - landmarks.part(54).y) + 0.1
    left_12 = np.abs(landmarks.part(33).y - landmarks.part(48).y) + 0.1


    if (np.mean(points_12_right)/right_12 > au_12_high):# and (np.mean(points_12_middle)/middle_12 < au_12_middle):
        list_of_au_current_frame += dict_["AU_12"]
        if np.mean(points_12_left)/left_12 > au_12_high:
            list_of_au_current_frame += dict_["AU_12_high"]
        elif np.mean(points_12_left)/left_12 > au_12_low:
            list_of_au_current_frame += dict_["AU_12_asym"]
        else:
            list_of_au_current_frame += dict_["AU_12_uni"]

    elif (np.mean(points_12_right)/right_12 > au_12_low): #and (np.mean(points_12_middle)/middle_12 < au_12_middle):
        list_of_au_current_frame += dict_["AU_12"]
        if np.mean(points_12_left)/left_12 > au_12_high:
            list_of_au_current_frame += dict_["AU_12_asym"]
        elif np.mean(points_12_left)/left_12 > au_12_low:
            list_of_au_current_frame += dict_["AU_12_low"]
        else:
            list_of_au_current_frame += dict_["AU_12_uni"]

    elif (np.mean(points_12_left)/left_12 > au_12_low): #and (np.mean(points_12_middle)/middle_12 < au_12_middle):
        list_of_au_current_frame += dict_["AU_12"]
        list_of_au_current_frame += dict_["AU_12_uni"]

    elif np.mean(points_10)/(landmarks.part(33).y -(landmarks.part(50).y +  landmarks.part(52).y)/2) > au_10:
        list_of_au_current_frame += dict_["AU_10"]

    points_10 = get_window(window_parameter, points_10, (landmarks.part(33).y -(landmarks.part(50).y +  landmarks.part(52).y)/2))
    points_12_right = get_window(window_parameter, points_12_right, right_12)
    points_12_left = get_window(window_parameter, points_12_left, left_12)
    # print(points_12_right, "points_12")

    return points_10, points_12_right, points_12_left, points_12_middle, list_of_au_current_frame


def AU_17(gray, landmarks, gray_scale_gabor_au_17, window_parameter, au_17, list_of_au_current_frame):
    gabor_kernel_1 = cv2.getGaborKernel((5, 5), 15.0, 220, 10.0, 0.5, psi = np.pi*0.5, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel_1)

    filtered_img = filtered_img[landmarks.part(57).y : landmarks.part(8).y,
                                landmarks.part(7).x : landmarks.part(9).x]

    if np.mean(gray_scale_gabor_au_17)/filtered_img.mean() < au_17:
        list_of_au_current_frame += dict_["AU_17"]
    gray_scale_gabor_au_17 = get_window(window_parameter, gray_scale_gabor_au_17, filtered_img.mean())

    return gray_scale_gabor_au_17, list_of_au_current_frame


def AU_24(landmarks, points_24, window_parameter, au_24, list_of_au_current_frame):
    diff_ = landmarks.part(50).y+landmarks.part(52).y - landmarks.part(56).y-landmarks.part(58).y
    if np.mean(points_24)/(diff_) > au_24:
        list_of_au_current_frame += dict_["AU_24"]
    points_24 = get_window(window_parameter, points_24, diff_)

    return points_24, list_of_au_current_frame



def AU_25_26(landmarks, points_25, points_26, window_parameter, au_26, au_25, list_of_au_current_frame):
    diff_26 = landmarks.part(33).y - np.mean([landmarks.part(56).y, landmarks.part(58).y])
    if np.mean(points_26)/(diff_26) < au_26:
        list_of_au_current_frame += dict_["AU_26"]

    diff_25 = landmarks.part(50).y+landmarks.part(52).y - landmarks.part(56).y-landmarks.part(58).y
    if np.mean(points_25)/(diff_25) < au_25:
        list_of_au_current_frame += dict_["AU_25"]

    points_26 = get_window(window_parameter, points_26, diff_26)
    points_25 = get_window(window_parameter, points_25, diff_25)

    return points_25, points_26, list_of_au_current_frame




def AU_43_45(landmarks, COUNTER, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES_BLINK, EYE_AR_CONSEC_FRAMES_CLOSE, fast_blinking, gazing, window_parameter_fast_blink_gaze, list_of_au_current_frame, norm_blinking_freq):
    left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
    right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

    if blinking_ratio < EYE_AR_THRESH:
        COUNTER += 1
        fast_blinking = get_window(window_parameter_fast_blink_gaze, fast_blinking, 0)
        gazing = get_window(window_parameter_fast_blink_gaze, gazing, 1)
        if COUNTER >= EYE_AR_CONSEC_FRAMES_CLOSE:
            list_of_au_current_frame += dict_["CLOSE"]

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

    return ver_, hor_, list_of_au_current_frame




def AU_M67_M68_75_76(gray, landmarks, au_list_eye, ratio_1_left, ratio_2_up, ratio_1_right, ratio_2_down,
    window_parameter_eye, window_parameter_quick_shift, list_of_au_current_frame, quick_shift, eye_, dict_poitns_eye, eye_text,
    points_area, window_parameter, area_parameter):

    eye_region, p1, p2, p3, p4, p5, p6 = eye_reg(landmarks, eye_, dict_poitns_eye)
    r_min_x = np.min(eye_region[:, 0])
    r_max_x = np.max(eye_region[:, 0])
    r_min_y = np.min(eye_region[:, 1])
    r_max_y = np.max(eye_region[:, 1])


    roi = gray[r_min_y: r_max_y, r_min_x+3: r_max_x-3]
    roi = cv2.equalizeHist(roi)
    # eye_thresh = np.median(roi.reshape(-1))
    eye_thresh = np.percentile(roi.reshape(-1), 15)
    # rows, cols = roi.shape
    gray_roi = cv2.GaussianBlur(roi, (7, 7), 0)
    _, threshold = cv2.threshold(gray_roi, eye_thresh, 255, cv2.THRESH_BINARY_INV)
    _, contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(_, key=lambda x: cv2.contourArea(x), reverse=True)

    # Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        area = w*h

        if (area + 1)/(np.mean(points_area) + 1) > area_parameter:
            list_of_au_current_frame += dict_["AU_32"]
        points_area = get_window(window_parameter, points_area, area)


        cv2.rectangle(gray, (r_min_x + x, r_min_y + y), (r_min_x + x + w, r_min_y + y + h), (255, 255, 255), 3)
        centre_ = (int((x + x + w)/2), int((y + y + h)/2))
        centre_ = (int(r_min_x+3 + centre_[0]), int(r_min_y + centre_[1]))

        # distance between eye left point and centre
        dist_1, _ = get_distance_points([landmarks.part(p1).x, landmarks.part(p1).y], centre_)
        # distance between eye left and right points
        dist_2, _ = get_distance_points([landmarks.part(p1).x, landmarks.part(p1).y], [landmarks.part(p4).x, landmarks.part(p4).y])

        ratio_1 = (dist_1)/(dist_2)
        # print(p1, p4,"p1p4")
        # print("p1_centre", [landmarks.part(p1).x, landmarks.part(p1).y], centre_)
        # print([landmarks.part(p1).x, landmarks.part(p1).y],[landmarks.part(p4).x, landmarks.part(p4).y])
        # print(dist_1, "dist_1")
        # print(dist_2,"dist_2")



        # distance between eye highest point and centre
        _, dist_3 = get_distance_points([(landmarks.part(p2).x + landmarks.part(p3).x)/2, (landmarks.part(p2).y + landmarks.part(p3).y)/2], centre_)

        # distance between eye highest point and lowest point

        _, dist_4 = get_distance_points([(landmarks.part(p2).x + landmarks.part(p3).x)/2, (landmarks.part(p2).y + landmarks.part(p3).y)/2],
                                    [(landmarks.part(p5).x + landmarks.part(p6).x)/2, (landmarks.part(p5).y + landmarks.part(p6).y)/2])
        if dist_4==0:
            dist_4 = 0.001

        ratio_2 = (dist_3)/(dist_4)

        # print("ratio_2", ratio_2,"____", ratio_2_down,  ratio_2_up)
        # if ratio_2>ratio_2_up and ratio_2 < ratio_2_down:
        #     print("Ratio 2    ok")
        # else:
        #     print("Ratio 2  not ok")
        #
        # print("ratio_1", ratio_1,"----", ratio_1_left, ratio_1_right)
        # if ratio_1>ratio_1_left and ratio_1 < ratio_1_right:
        #     print("Ratio 1 ok")
        # else:
        #     print("Ratio 1 not ok")

        if ratio_1 < ratio_1_left:
            if ratio_2 < ratio_2_up:
                au_list_eye = get_window(window_parameter_eye,au_list_eye, eye_text["eye_left_up"])
                list_of_au_current_frame += dict_["eye_au_m67"]
            elif ratio_2 > ratio_2_down:
                au_list_eye = get_window(window_parameter_eye,au_list_eye, eye_text["eye_left_down"])
                list_of_au_current_frame += dict_["eye_au_75"]
            else:
                au_list_eye = get_window(window_parameter_eye,au_list_eye, eye_text["eye_left_middle"])
                list_of_au_current_frame += dict_["eye_left_middle"]
                quick_shift = get_window(window_parameter_quick_shift, quick_shift, eye_text["eye_left_middle"])
        elif ratio_1 > ratio_1_right:
            if ratio_2 < ratio_2_up:
                au_list_eye = get_window(window_parameter_eye,au_list_eye,eye_text["eye_right_up"])

                list_of_au_current_frame += dict_["eye_au_m68"]
            elif ratio_2 > ratio_2_down:
                au_list_eye = get_window(window_parameter_eye, au_list_eye,eye_text["eye_right_down"])

                list_of_au_current_frame += dict_["eye_au_76"]
            else:
                au_list_eye = get_window(window_parameter_eye, au_list_eye, eye_text["eye_right_middle"])

                list_of_au_current_frame += dict_["eye_right_middle"]
                quick_shift = get_window(window_parameter_quick_shift, quick_shift, eye_text["eye_right_middle"])
        else:
            if ratio_2 < ratio_2_up:
                au_list_eye = get_window(window_parameter_eye,au_list_eye,eye_text["eye_middle_up"])
                list_of_au_current_frame += dict_["eye_middle_up"]
            elif ratio_2 > ratio_2_down:
                au_list_eye = get_window(window_parameter_eye,au_list_eye,eye_text["eye_middle_down"])
                list_of_au_current_frame += dict_["eye_middle_down"]
            else:
                au_list_eye = get_window(window_parameter_eye, au_list_eye, eye_text["eye_middle_middle"])
                list_of_au_current_frame += dict_["eye_middle_middle"]
                # print(dict_["eye_middle_middle"])
                # print(list_of_au_current_frame, "close eye" )

        # if sorted(set(au_list_eye), key=au_list_eye.index)==[eye_text["eye_middle_middle"], eye_text["eye_middle_up"], eye_text["eye_left_middle"]] and eye_text["eye_middle_middle"] in au_list_eye[au_list_eye.index(eye_text["eye_left_middle"]):]:
        #     list_of_au_current_frame += dict_["eye_au_m67"]
        # elif sorted(set(au_list_eye), key=au_list_eye.index)==[eye_text["eye_middle_middle"], eye_text["eye_middle_up"],eye_text["eye_right_middle"]] and eye_text["eye_middle_middle"] in au_list_eye[au_list_eye.index(eye_text["eye_right_middle"]):]:
        #     list_of_au_current_frame += dict_["eye_au_m68"]
        # elif sorted(set(au_list_eye), key=au_list_eye.index)==[eye_text["eye_middle_middle"], eye_text["eye_middle_down"],eye_text["eye_left_middle"]] and eye_text["eye_middle_middle"] in au_list_eye[au_list_eye.index(eye_text["eye_left_middle"]):]:
        #     list_of_au_current_frame += dict_["eye_au_75"]
        # elif sorted(set(au_list_eye), key=au_list_eye.index)==[eye_text["eye_middle_middle"], eye_text["eye_middle_down"],eye_text["eye_right_middle"]] and eye_text["eye_middle_middle"] in au_list_eye[au_list_eye.index(eye_text["eye_right_middle"]):]:
        #     list_of_au_current_frame += dict_["eye_au_76"]
        # break

        #if sorted(set(au_list_eye), key=au_list_eye.index)==[eye_text["eye_left_middle"], eye_text["eye_left_up"]]:
        #    if eye_text["eye_left_middle"] in au_list_eye[au_list_eye.index(eye_text["eye_left_up"]):]:
        #        list_of_au_current_frame += dict_["Combo_49_b"]
        #    elif eye_text["eye_right_up"] in au_list_eye[au_list_eye.index(eye_text["eye_left_up"]):]:
        #        index_ = au_list_eye.index(eye_text["eye_right_up"])
        #        if sorted(set(au_list_eye[index_:]), key=au_list_eye[:index_].index) == [eye_text["eye_left_up"], eye_text["eye_left_middle"]]:
        #            list_of_au_current_frame += dict_["Combo_49_b"]
        #        elif eye_text["eye_right_middle"] in au_list_eye[index_:]:
        #            list_of_au_current_frame += dict_["Combo_49_b"]

        #elif sorted(set(au_list_eye), key=au_list_eye.index)==[eye_text["eye_right_middle"], eye_text["eye_right_up"]]:
        #    if eye_text["eye_right_middle"] in au_list_eye[au_list_eye.index(eye_text["eye_right_up"]):]:
        #        list_of_au_current_frame += dict_["Combo_49_b"]
        #    elif eye_text["eye_left_up"] in au_list_eye[au_list_eye.index(eye_text["eye_right_up"]):]:
        #        index_ = au_list_eye.index(eye_text["eye_left_up"])
        #        if sorted(set(au_list_eye[index_:]), key=au_list_eye[:index_].index) == [eye_text["eye_right_up"], eye_text["eye_right_middle"]]:
        #            list_of_au_current_frame += dict_["Combo_49_b"]
        #        elif eye_text["eye_left_middle"] in au_list_eye[index_:]:
        #            list_of_au_current_frame += dict_["Combo_49_b"]
        
        if set(sorted(set(au_list_eye), key=au_list_eye.index)).intersection(set([eye_text["eye_right_up"], eye_text["eye_left_up"]]))==set([eye_text["eye_right_up"], eye_text["eye_left_up"]]):
            list_of_au_current_frame += dict_["Combo_49_b"]
        elif set(sorted(set(au_list_eye), key=au_list_eye.index)).intersection(set([eye_text["eye_left_up"], eye_text["eye_right_up"]]))==set([eye_text["eye_left_up"], eye_text["eye_right_up"]]):
            list_of_au_current_frame += dict_["Combo_49_b"]

        break

    if sorted(set(quick_shift), key=quick_shift.index)==[eye_text["eye_left_middle"], eye_text["eye_right_middle"]] and (eye_text["eye_left_middle"] in quick_shift[quick_shift.index(eye_text["eye_right_middle"]):]):
        list_of_au_current_frame += dict_["quick_shift_"]
    elif sorted(set(quick_shift), key=quick_shift.index)==[eye_text["eye_right_middle"], eye_text["eye_left_middle"]] and (eye_text["eye_right_middle"] in quick_shift[quick_shift.index(eye_text["eye_left_middle"]):]):
        list_of_au_current_frame += dict_["quick_shift_"]
    return au_list_eye, list_of_au_current_frame, quick_shift, points_area


def head_movement(window_parameter, landmarks, x_coord, y_coord, x_coord_thresh, y_coord_thresh):
    ver_move = False
    hor_move = False
    x_diff = np.mean(x_coord)/landmarks.part(33).x
    if x_diff < x_coord_thresh[0] or x_diff > x_coord_thresh[1]:
        ver_move = True

    y_diff = np.mean(x_coord)/landmarks.part(33).y
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

