import numpy as np
import cv2
from processing import *
from utils import *
from sudoku_solve import Sudoku_solver


input_str = "000000000008236400010050020500000009100000007080000050005000200000807000000020000"
def draw_grid():
    base_img = 1* np.ones((600,600))
    width = base_img.shape[0] // 9
    cv2.rectangle(base_img, (0,0), (base_img.shape[0], base_img.shape[1]), (0,0,0), 10)
    for i in range(1,10):
        cv2.line(base_img, (i*width, 0), (i*width, base_img.shape[1]), (0,0,0), 5)
        cv2.line(base_img, (0, i* width), (base_img.shape[0], i*width), (0,0,0), 5)
    return base_img

def draw_digit(base_img, input_str):
    width = base_img.shape[0] // 9
    board = convert_str_to_board(input_str)
    for j in range(9):
        for i in range(9):
            if board[j][i] !=0 : # Only draw new number to blank cell in warped image, avoid overlapping 

                p1 = (i * width, j * width)  # Top left corner of a bounding box
                p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

                # Find the center of square to draw digit
                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(board[j][i]), cv2.FONT_HERSHEY_SIMPLEX, 1, 6)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(base_img, str(board[j][i]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
    return base_img, board

def solve(board):
    unsolved_board = board.copy()
    sudoku = Sudoku_solver(board, 9)
    sudoku.solve()
    res_board = sudoku.board
    return res_board, unsolved_board


def draw_result(base_img, solved_board):
    width = base_img.shape[0] // 9
    for j in range(9):
        for i in range(9):  
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

            # Find the center of square to draw digit
            center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            text_size, _ = cv2.getTextSize(str(solved_board[j][i]), cv2.FONT_HERSHEY_SIMPLEX, 1, 6)
            text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
            cv2.putText(base_img, str(solved_board[j][i]),
                        text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
    return base_img

base_img = draw_grid()
res_img = base_img.copy()
base_img, board = draw_digit(base_img, input_str)
cv2.imshow("IMG", base_img)
cv2.waitKey(0)

res_board, unsolved_board = solve(board)
res_img = draw_result(res_img, unsolved_board, res_board)
cv2.imshow("Show result", res_img)
cv2.waitKey(0)

