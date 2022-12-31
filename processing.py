import cv2
import numpy as np
from threshold import preprocess
from helper_module import find_corners, draw_circle_at_corners, grid_line_helper, draw_line
from helper_module import clean_square_helper, classify_one_digit

def find_contours(img, original):
    """
    contours: A tuple of all point creating contour lines, each contour is a np array of points (x,y).
    hierachy: [Next, Previous, First_Child, Parent]
    contour approximation: https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
    """
    # find contours on threshold image
    contours, hierachy =  cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort the largest contour to find the puzzle
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    polygon = None
    # find the largest rectangle-shape contour to make sure this is the puzzle
    for con in contours:
        area = cv2.contourArea(con)
        perimeter = cv2.arcLength(con, closed = True)
        approx = cv2.approxPolyDP(con, epsilon=0.01 * perimeter, closed =  True)
        num_of_ptr = len(approx)
        if num_of_ptr == 4 and area > 1000:
            polygon = con   #finded puzzle
            break
    if polygon is not None:
        # find corner
        top_left = find_corners(polygon, limit_func= min, compare_func= np.add)
        top_right = find_corners(polygon, limit_func= max, compare_func= np.subtract)
        bot_left = find_corners(polygon,limit_func=min, compare_func= np.subtract)
        bot_right = find_corners(polygon,limit_func=max, compare_func=np.add)
        #Check polygon is square, if not return []
        #Set threshold rate for width and height
        if not (0.8 < ((top_right[0]-top_left[0]) / (bot_right[1]-top_right[1]))<1.2):
            print("Exception 1 : Get another image to get square-shape puzzle")
            return [],[]
        if bot_right[1] - top_right[1] == 0:
            print("Exception 2 : Get another image to get square-shape puzzle")
            return [],[]
        corner_list = [top_left, top_right, bot_left, bot_right]
        cv2.drawContours(original, [polygon], 0, (255,255,0), 3)
        #draw circle at each corner point
        for x in corner_list:
            draw_circle_at_corners(original, x)

        return original, corner_list
    print("Can not detect puzzle")
    return [],[]



def warp_image(corner_list, original):
    """
    Input: 4 corner points and threshold grayscale image
    Output: Perspective transformation matrix and transformed image
    Perspective transformation: https://theailearner.com/tag/cv2-warpperspective/
    """
    corners = np.array(corner_list, dtype= "float32")
    top_left, top_right, bot_left, bot_right = corners[0], corners[1], corners[2], corners[3]
    #Get the largest side to be the side of squared transfromed puzzle
    side = int(max([
        np.linalg.norm(top_right - bot_right),
        np.linalg.norm(top_left - bot_left),
        np.linalg.norm(bot_right - bot_left),
        np.linalg.norm(top_left - top_right)
    ]))
    out_ptr = np.array([[0,0],[side-1,0],[0,side-1],[side-1,side-1]],dtype="float32")
    transfrom_matrix = cv2.getPerspectiveTransform(corners, out_ptr)
    transformed_image = cv2.warpPerspective(original, transfrom_matrix, (side, side))
    return transformed_image, transfrom_matrix



def get_grid_line(img, length = 10):
    """
    Get horizontal and vertical lines from warped image
    """
    horizontal = grid_line_helper(img, shape_location= 1)
    vertical = grid_line_helper(img, shape_location=0)
    return vertical, horizontal



def create_grid_mask(vertical, horizontal):
    """
    Completely detect all lines by using Hough Transformation
    Create grid mask to extract number by using bitwise_and with warped images
    """
    # combine two line to make a grid
    grid = cv2.add(horizontal, vertical)
    # Apply threshold to cover more area
    # grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    morpho_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    grid = cv2.dilate(grid, morpho_kernel, iterations=2)
    # find the line by Houghline transfromation
    lines = cv2.HoughLines(grid, 0.3, np.pi/90, 200)
    lines_img = draw_line(grid, lines)
    # Extract all the lines
    mask = cv2.bitwise_not(lines_img)
    return mask



def split_squares(number_img):
    """
    Split number img into 81 squares.
    """
    square_list = []
    side = number_img.shape[0] // 9
    #find each square and append to square_list
    for j in range(0,9):
        for i in range(0,9):
            top_left_square = (i * side, j * side)
            bot_right_square = ((i+1) * side, (j+1) * side)
            square_list.append(number_img[top_left_square[1]:bot_right_square[1], top_left_square[0]: bot_right_square[0]])

    return square_list



def clean_square(square_list):
    """
    Return cleaned-square list and number of digits available in the image
    """
    cleaned_squares = []
    count = 0
    for sq in square_list:
        new_img, is_num = clean_square_helper(sq)
        if is_num:
            cleaned_squares.append(new_img)
            count += 1
        else:
            cleaned_squares.append(0)
    return cleaned_squares, count

def recognize_digits(model, resized):
    res_str = ""
    for img in resized:
        digit = classify_one_digit(model, img, threshold=80)
        res_str += str(digit)
    return res_str

if __name__ == "__main__":
    img2 = "Testimg\sudoku.jpg"
    img2 = cv2.imread(img2)
    thresholded = preprocess(img2)
    res, corners = find_contours(thresholded, thresholded)
    res_img, matrix = warp_image(corners, res)
    res_img = cv2.resize(res,(600,600) ,interpolation = cv2.INTER_AREA)
    cv2.imshow("Img", res_img)
    cv2.waitKey(0)
    print(res_img.shape)
    print(corners)
    print(matrix)