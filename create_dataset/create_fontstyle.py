import glob
import random
from PIL import Image, ImageDraw, ImageFont

font_folder = 'font'
font_name = ['arial', 'bodoni','calibri','futura','heveltica','times-new-roman']

def fontstyle_list(font_folder, font_name):
    font_list = []
    for i in font_name:
        font_dir = glob.glob(font_folder + "\\"+ i +"\\*.ttf")
        for j in font_dir:
            font_list.append(j)
    return font_list



def draw_img(label, font_list):
    img = Image.new('L', (256, 256))
    size = random.randint(150, 250)
    x = random.randint(60, 90)
    y = random.randint(30, 60)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(, )
    font = ImageFont.truetype(font_list[0], size)
    draw.text((x, y), str(label), (200),font=font)

    img = img.resize((28, 28), Image.BILINEAR)
    return img, label

if __name__ == "__main__":
    fonts = fontstyle_list(font_folder, font_name)
    print(fonts)
    print(len(fonts))