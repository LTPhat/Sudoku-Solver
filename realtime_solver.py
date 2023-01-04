import numpy as np
import torch
from utils import *
from processing import * 
from threshold import preprocess
import time
import cv2
from sudoku_solve import Sudoku_solver


classifier = torch.load('digit_classifier.h5',map_location ='cpu')
classifier.eval()


frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

prev = 0

# while True:
#     time_elapsed = time.time() - prev
#     success, img = cap.read()
#     if time_elapsed > 1. / frame_rate:
#         prev = time.time()
#         final_img = img.copy()
#         to_process_img = img.copy()
#         #Processing 
#         thresholded_img = preprocess(to_process_img) # Gray-scale img
#         corners_img, corner_list, org_img = find_contours(thresholded_img, to_process_img)

#         if corner_list:


#     # cv2.imshow("IMG", img)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break
# cv2.destroyAllWindows()
# cap.release()