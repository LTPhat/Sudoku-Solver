from PIL import Image, ImageFont, ImageDraw
import random
import cv2
import numpy as np

#font size
fonts = ['font\Arial\Arial.ttf', 'font\calibri\Calibri-Bold.ttf','font\heveltica\helvetica_bold.ttf','font\Times-new-roman\Times.ttf']

img = Image.new('L', (256, 256))

target = random.randint(0, 9)

size = random.randint(150, 250)
x = random.randint(60, 90)
y = random.randint(30, 60)
draw = ImageDraw.Draw(img)
# font = ImageFont.truetype(, )
font = ImageFont.truetype(fonts[random.randint(0,3)], size)
draw.text((x, y), str(target), (200),font=font)

img = img.resize((28, 28), Image.BILINEAR)
img = np.array(img)
cv2.imshow("Image", img)
cv2.waitKey(0)