import glob


font_folder = 'font'
font_name = ['arial', 'bodoni','calibri','futura','heveltica','times-new-roman']

def fontstyle_list(font_folder, font_name):
    font_list = []
    for i in font_name:
        font_dir = glob.glob(font_folder + "\\"+ i +"\\*.ttf")
        for j in font_dir:
            font_list.append(j)
    return font_list

if __name__ == "__main__":
    fonts = fontstyle_list(font_folder, font_name)
    print(fonts)
    print(len(fonts))