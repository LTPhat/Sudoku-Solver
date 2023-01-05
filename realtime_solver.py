import numpy as np
import torch
from utils import *
from processing import * 
from threshold import preprocess
import time
import cv2
from sudoku_solve import Sudoku_solver
from PIL import Image

classifier = torch.load('digit_classifier.h5',map_location ='cpu')
classifier.eval()


frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(0)
frame_rate = 60

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness 
cap.set(10, 100)
prev = 0

while True:
    time_elapsed = time.time() - prev
    success, img = cap.read()
    if time_elapsed > 1. / frame_rate:
        prev = time.time()
        final_img = img.copy()
        to_process_img = img.copy()
        #Processing 
        thresholded_img = preprocess(to_process_img) # Gray-scale img
        corners_img, corners_list, org_img = find_contours(thresholded_img, to_process_img)

        if corners_list:
            # Warped original img 
            warped, matrix = warp_image(corners_list, corners_img)
            # Threshold warped img
            warped_processed = preprocess(warped) # warped_processed is gray-scaled img

            #Get lines
            horizontal = grid_line_helper(warped_processed, shape_location=0)
            vertical = grid_line_helper(warped_processed, shape_location=1)

            # Create mask
            grid_mask = create_grid_mask(horizontal, vertical)
            # Resize will get better result ??
            grid_mask = cv2.resize(grid_mask,(600,600), cv2.INTER_AREA)
            # Extract number
            number_img = cv2.bitwise_and(cv2.resize(warped_processed, (600,600), cv2.INTER_AREA), grid_mask)
            # number_img = cv2.bitwise_and(warped_processed, grid_mask)
            # Split into squares
            squares = split_squares(number_img)
            cleaned_squares = clean_square_all_images(squares)

            # Resize and scale pixel
            resized_list = resize_square(cleaned_squares)
            norm_resized = normalize(resized_list)

            # # Recognize digits
            rec_str = recognize_digits(classifier, norm_resized)
            board = convert_str_to_board(rec_str)
            
            # Solve
            unsolved_board = board.copy()
            sudoku = Sudoku_solver(board, 9)
            start_time = time.time()
            sudoku.solve()
            solved_board = sudoku.board
            # Unwarp
            _, warp_with_nums = draw_digits_on_warped(warped, solved_board, unsolved_board)

            final_img = unwarp_image(warp_with_nums, corners_img, corners_list, time.time() - start_time)
            cv2.imshow("Result", final_img)
            # final_img.save("{}.png".format(count))

        else:
            cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()