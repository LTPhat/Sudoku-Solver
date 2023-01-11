import cv2
import numpy as np
from threshold import preprocess
from processing import find_contours, warp_image, create_grid_mask, split_squares, clean_square, recognize_digits, draw_digits_on_warped, unwarp_image
from utils import grid_line_helper, clean_square_helper, resize_square, classify_one_digit, normalize, convert_str_to_board
import matplotlib.pyplot as plt
import torch 
from sudoku_solve import Sudoku_solver



classifier = torch.load('digit_model.h5',map_location ='cpu')
classifier.eval()


img = "streamlit_app\image_from_user\Real_test2.jpg"
img = cv2.imread(img)
thresholded = preprocess(img)
corners_img, corners, org_img = find_contours(thresholded, img)
warped, matrix = warp_image(corner_list=corners, original= corners_img)
warped_processed = preprocess(warped)
horizontal = grid_line_helper(warped_processed, shape_location = 0)
vertical = grid_line_helper(warped_processed, shape_location=1)



def test_wrap_image(thresholded):
    corners_img, corners,_ = find_contours(thresholded, thresholded)
    res_img, matrix = warp_image(corners, corners_img)
    res_img = cv2.resize(res_img, (600,600),interpolation = cv2.INTER_AREA)
    cv2.imshow("Wraped image", res_img)
    cv2.waitKey(0)
    print(matrix)
    print("Test wrap image success")



def test_threshold_img(original):
    thresholded = preprocess(original)
    cv2.imshow("img", thresholded)
    cv2.waitKey(0)
    print("Test threshold image success")
    return thresholded



def test_find_contours(thresholded, original):
    original, corner_list = find_contours(thresholded, thresholded)
    cv2.imshow("img", original)
    print(corner_list)
    print("Test find contours success")
    cv2.waitKey(0)



def test_draw_contour(thresholded, original):
    contour, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original, contour, -1, (0,255,255), 3)
    original = cv2.resize(original, (600,600), interpolation= cv2.INTER_AREA)
    cv2.imshow("Draw Contour", original)
    cv2.waitKey(0)



def test_get_gridline(original, length = 10):
    horizontal = grid_line_helper(original, shape_location =0)
    vertical = grid_line_helper(original, shape_location=1)
    horizontal = cv2.resize(horizontal, (600,600), interpolation= cv2.INTER_AREA)
    cv2.imshow("Original", original)
    cv2.waitKey(0)
    cv2.imshow("Find horizontal grid line", horizontal)
    cv2.waitKey(0)
    vertical = cv2.resize(vertical, (600,600), interpolation= cv2.INTER_AREA)
    cv2.imshow("Find vertical grid line", vertical)
    cv2.waitKey(0)
    print("Test get gridline success")



def test_create_grid_mask(horizontal, vertical):
    get = create_grid_mask(horizontal, vertical)
    get = cv2.resize(get, (600,600), interpolation= cv2.INTER_AREA)
    cv2.imshow("Mask", get)
    cv2.waitKey(0)
    print("Test get gridline success")
    return get



def test_split_square(number_img):
    square = split_squares(number_img)
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 9, 9
    plt.title("Split into 81 squares")
    plt.axis("off")
    #Visualize result
    for i in range(0, cols * rows):
        figure.add_subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(square[i], cmap="gray")
    plt.show()
    print("Test split square success")
    return square



#Test clean output image
def test_clean_square_visualize(square_list):
    square_cleaned_list = []
    for i in square_list:
        clean_square, _ = clean_square_helper(i)
        square_cleaned_list.append(clean_square)
    figure = plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Clean noises in square images")
    cols, rows = 9, 9
    #Visualize result
    for i in range(0, cols * rows):
        figure.add_subplot(rows, cols, i+1)
        plt.axis("off")
        plt.imshow(square_cleaned_list[i], cmap="gray")
    plt.show()
    print("Test clean square success")
    return square_cleaned_list




def test_clean_square_count(square_list):
    cleaned_list, count = clean_square(square_list)
    print(count)
    print("Test clean quare count success")



def test_resize_clean_square(clean_square_list):
    resized_list = resize_square(clean_square_list)
    for i in range(0,5):
        print(resized_list[i].shape)
    print("Test resize clean square success")
    return resized_list



def test_classify_one_digit(model, resize_list):
    digit = classify_one_digit(model, clean_square, threshold=60)
    print("Test classify one digit success")
    return digit



def test_recognize_digits(model, resize_list, org_image):
    res_str = recognize_digits(model, resize_list, org_image)
    return res_str



def test_convert_str_to_board(string):
    board = convert_str_to_board(string)
    # print(board)
    # print(type(board))
    print("Test convert str to board success")
    return board



def test_sudoku_solver(board):
    unsolved_board = board.copy()
    sudoku = Sudoku_solver(board, 9)
    # sudoku.print_board()
    sudoku.solve()
    # sudoku.print_board()
    res_board = sudoku.board
    print("Test sudoku solver success")
    return res_board, unsolved_board



def test_draw_digits_warped(warped_img, solved_board, unsolved_board):
    img_text, warped_img= draw_digits_on_warped(warped_img, solved_board, unsolved_board)
    # img_text = cv2.resize(img_text, (600,600), interpolation=cv2.INTER_AREA)
    return img_text, warped_img


def test_unwarp_image(img_src, img_dst, corner_list):
    dst_img = unwarp_image(img_src, img_dst, corner_list, 0.115)
    # cv2.imshow("Res", dst_img)
    # cv2.waitKey(0)
    return dst_img

if __name__ == "__main__":

    cv2.imshow("Original image", cv2.resize(img, (600,600)))
    cv2.waitKey(0)
    cv2.imshow("Find corners", cv2.resize(corners_img, (600,600), cv2.INTER_AREA))
    cv2.waitKey(0)
    get = test_create_grid_mask(horizontal, vertical)
    number = cv2.bitwise_and(cv2.resize(warped_processed, (600,600), cv2.INTER_AREA), get)
    square = test_split_square(number)
    square_cleaned_list = test_clean_square_visualize(square)
    resized = test_resize_clean_square(square_cleaned_list)
    resize_norm = normalize(resized)
    res_str = test_recognize_digits(classifier, resize_norm, img)
    print(res_str)

    board = test_convert_str_to_board(res_str)
    res_board, unsolved_board = test_sudoku_solver(board)
    print(res_board)
    print(unsolved_board)

    _, warp_with_nums = test_draw_digits_warped(warped, res_board, unsolved_board)
    cv2.imshow("Warped with numbers", cv2.resize(warp_with_nums, (600,600), cv2.INTER_AREA))
    cv2.waitKey(0)
    print(warp_with_nums.shape)
    dst_img = test_unwarp_image(warp_with_nums, img, corners)
    cv2.imshow("Final result", cv2.resize(dst_img, (800,800), cv2.INTER_AREA))
    cv2.waitKey(0)
