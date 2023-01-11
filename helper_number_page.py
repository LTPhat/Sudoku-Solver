import numpy as np
import cv2
from processing import *
from utils import *
from sudoku_solve import Sudoku_solver


input_str = "000000000008236400010050020500000009100000007080000050005000200000807000000020000"
input_str2 = "800010009050807010004090700060701020508060107010502090007040600080309040300050008"

def draw_grid():
    base_img = 1* np.ones((600,600,3))
    width = base_img.shape[0] // 9
    cv2.rectangle(base_img, (0,0), (base_img.shape[0], base_img.shape[1]), (0,0,0), 10)
    for i in range(1,10):
        if i % 3 == 0:
            cv2.line(base_img, (i*width, 0), (i*width, base_img.shape[1]), (0,0,0), 6)
            cv2.line(base_img, (0, i* width), (base_img.shape[0], i*width), (0,0,0), 6)
        else:
            cv2.line(base_img, (i*width, 0), (i*width, base_img.shape[1]), (0,0,0), 2)
            cv2.line(base_img, (0, i* width), (base_img.shape[0], i*width), (0,0,0), 2)
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
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
    return base_img, board

def solve(board):
    unsolved_board = board.copy()
    sudoku = Sudoku_solver(board, 9)
    sudoku.solve()
    res_board = sudoku.board
    return res_board, unsolved_board


def draw_result(base_img, unsolved_board, solved_board):
    width = base_img.shape[0] // 9
    for j in range(9):
        for i in range(9):  
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

            # Find the center of square to draw digit
            center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            text_size, _ = cv2.getTextSize(str(solved_board[j][i]), cv2.FONT_HERSHEY_SIMPLEX, 1, 6)
            text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
            if unsolved_board[j][i] != solved_board[j][i]:
                cv2.putText(base_img, str(solved_board[j][i]),
                        text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)
            else:
                cv2.putText(base_img, str(solved_board[j][i]),
                        text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
    return base_img

### CHECK VALID SODOKU PUZZLE INPUT FROM USER
def get_column(board, index):
    return np.array([row[index] for row in board])


def valid_row_or_col(array):
    if np.all(array == 0) == True:
        return True
    return len(set(array[array!=0])) == len(list(array[array!=0]))

def valid_single_box(board, box_x, box_y):
    box = board[box_x*3 : box_x*3 + 3, box_y*3: box_y*3+3]
    if len(list(box[box!=0])) == 0:
        return True
    return len(set(box[box!=0])) == len(list(box[box!=0]))

def valid_input_str(input_str):
    board = convert_str_to_board(input_str)
    # Check valid row
    for i in range(0,len(board)):
        if valid_row_or_col(board[i]) == False:
            return False
    # Check valid column
    for j in range(0, len(board[0])):
        if valid_row_or_col(get_column(board, j)) == False:
            return False
    # Check valid box
    for i in range(0, 3):
        for j in range(0, 3):
            if valid_single_box(board, i, j) == False:
                return False
    return True

def valid_board(board):
    # Check valid row
    for i in range(0,len(board)):
        if valid_row_or_col(board[i]) == False:
            return False
    # Check valid column
    for j in range(0, len(board[0])):
        if valid_row_or_col(get_column(board, j)) == False:
            return False
    # Check valid box
    for i in range(0, 3):
        for j in range(0, 3):
            if valid_single_box(board, i, j) == False:
                return False
    return True

if __name__ == "__main__":
    base_img = draw_grid()
    res_img = base_img.copy()
    base_img, board = draw_digit(base_img, input_str)
    cv2.imshow("IMG", base_img)
    cv2.waitKey(0)

    res_board, unsolved_board = solve(board)
    res_img = draw_result(res_img, unsolved_board, res_board)
    cv2.imshow("Show result", res_img)
    cv2.waitKey(0)
    res = valid_input_str(input_str2)
    print(res)
