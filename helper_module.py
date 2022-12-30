import cv2
import numpy as np
import operator

def find_corners(polygon, limit_func, compare_func):
    """
    Input: Rectangle puzzle extract from contours
    Output: One of four cornet point depend on limit_func, compare_func
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function
    Note: (0,0) point is at the top-left

    top-left: (x+y) min
    top-right: (x-y) max
    bot-left: (x-y) min
    bot-right: (x+y) max
    """
    index, _ = limit_func(enumerate([compare_func(ptr[0][0], ptr[0][1]) for ptr in polygon]), key = operator.itemgetter(1))

    return polygon[index][0][0], polygon[index][0][1]


def draw_circle_at_corners(original, ptr):
    """
    Helper function to draw circle at corners
    """
    cv2.circle(original, ptr, 15, (0,255,0), cv2.FILLED)


def grid_line_helper(img, shape_location, length = 10):
    """
    Helper function to fine vertical, horizontal line
    Find vertical line: shape_location = 0
    Find horizontal line: shape_location = 1
    """
    clone_img = img.copy()
    row_or_col = clone_img.shape[shape_location]

    # Find the distance between lines
    size = row_or_col // length

    # Morphological transfromation to find line

    # Define morphology kernel
    if shape_location == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size,1))

    clone_img = cv2.erode(clone_img, kernel)
    clone_img = cv2.dilate(clone_img, kernel)

    return clone_img
def draw_line(img, lines):
    """
    Draw all lines in lines got from cv2.HoughLine()
    """
    clone_img = img.copy()
    # lines list from cv2.HoughLine() is 3d array
    # Convert to 2d array

    lines = np.squeeze(lines)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        #Draw line every loop
        cv2.line(clone_img, (x1,y1), (x2,y2), (255,255,255), 4)
    return clone_img


def clean_square_helper(img):
    """
    Clean noises in every square splited
    Input: One of 81 squares
    Output: Cleaned square and boolean var which so that there is number in it
    """

    if np.isclose(img, 0).sum() / (img.shape[0] * img.shape[1]) >= 0.95:
        return np.zeros_like(img), False

    # if there is very little white in the region around the center, this means we got an edge accidently
    height, width = img.shape
    mid = width // 2
    if np.isclose(img[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.90:
        return np.zeros_like(img), False

    # center image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    start_x = (width - w) // 2
    start_y = (height - h) // 2
    new_img = np.zeros_like(img)
    new_img[start_y:start_y + h, start_x:start_x + w] = img[y:y + h, x:x + w]

    return new_img, True
