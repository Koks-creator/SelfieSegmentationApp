from typing import NamedTuple
from time import time
import mediapipe as mp
import cv2
import numpy as np


def get_landmarks(img: np.array, results: NamedTuple, draw=True) -> list:
    h, w, _ = img.shape
    landmark_list = []

    if results.pose_landmarks:
        if draw:
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for lm in results.pose_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_list.append((x, y, lm.z, lm.visibility))

    return landmark_list


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(enable_segmentation=True)
mp_draw = mp.solutions.drawing_utils

bg_img = cv2.imread(r"C:\Users\table\PycharmProjects\pajtong\SelfieSegmentationApp-main\Backgrounds\bg4.jpg")
p_time = 0

with pose:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        lm_list = get_landmarks(img, results)
        if lm_list:
            pass
            # print(lm_list)

            final_img = img.copy()
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_img = cv2.resize(bg_img, (w, h))

            final_img = np.where(condition, final_img, bg_img)

            c_time = time()
            fps = int(1 / (c_time - p_time))
            p_time = c_time
            cv2.putText(final_img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 100), 2)

            # cv2.imshow("Res", img)
            cv2.imshow("FinalImg", final_img)
        cv2.imshow("Res", img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
