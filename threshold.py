import cv2
import numpy as np


def preprocess(img):
    """
    Input: Original image
    Output: Gray-scale processed image
    """
    # convert RGB to gray-scale
    if (np.array(img).shape[2] != 1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gassian blur
    blured = cv2.GaussianBlur(gray_img, (9,9), 0)
    #set a threshold
    thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #invert so that the grid line and text are line, the rest is black
    inverted = cv2.bitwise_not(thresh, 0)
    morphy_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # Opening morphology to remove noise (while dot etc...)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, morphy_kernel)
    # dilate to increase border size
    result = cv2.dilate(morph, morphy_kernel, iterations=1)
    return result


if __name__ == "__main__":
    img = "Testimg\sudoku.jpg"
    img = cv2.imread(img)
    processed = preprocess(img)
    cv2.imshow("img", processed)
    cv2.waitKey(0)
