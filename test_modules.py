import cv2
import numpy as np
from threshold import preprocess
from processing import find_contours, warp_image, create_grid_mask, split_squares
from helper_module import grid_line_helper, clean_square_helper
import matplotlib.pyplot as plt

img = "Testimg\sudoku.jpg"
img = cv2.imread(img)
thresholded = preprocess(img)
corners_img, corners = find_contours(thresholded, img)
warped, matrix = warp_image(corner_list=corners, original= corners_img)
warped_processed = preprocess(warped)
horizontal = grid_line_helper(warped_processed, shape_location =0)
vertical = grid_line_helper(warped_processed, shape_location=1)



def test_wrap_image(thresholded):
    corners_img, corners = find_contours(thresholded, thresholded)
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

def test_clean_square(square_list):
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

if __name__ == "__main__":
    get = test_create_grid_mask(horizontal, vertical)
    number = cv2.bitwise_and(cv2.resize(warped_processed, (600,600), cv2.INTER_AREA), get)
    square = test_split_square(number)
    square_cleaned_list = test_clean_square(square)