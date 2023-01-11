import cv2
import numpy as np
from PIL import Image

# Reshape image for readme display
url1 = 'result_image_solver\Res1.png'
url2 = "result_image_solver\Res2.png"
url3 = "result_image_solver\Res3.png"

img1 = Image.open(url1)
img2 = Image.open(url2)
img3 = Image.open(url3)

img1 = np.array(img1)
img2 = np.array(img2)
img3 = np.array(img3)
print(img1.shape)
print(img2.shape)
print(img3.shape)

img1 = cv2.resize(img1, (500,500), cv2.INTER_AREA)
img2 = cv2.resize(img2, (500,500), cv2.INTER_AREA)
img3 = cv2.resize(img3, (500,500), cv2.INTER_AREA)

im1 = Image.fromarray(img1)
im2 = Image.fromarray(img2)
im3 = Image.fromarray(img3)

im1.save("result_image_solver\pic1.png")
im2.save("result_image_solver\pic2.png")
im3.save("result_image_solver\pic3.png")
