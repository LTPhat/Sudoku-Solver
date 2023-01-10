import numpy as np
import cv2
from processing import *
from utils import *
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import base64
import os
from PIL import Image
from image_solver import image_solver
from sudoku_solve import Sudoku_solver
import time
# Define model

classifier = torch.load('digit_classifier.h5',map_location ='cpu')
classifier.eval()


# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
# webrtc_streamer(
#     # ...
#     rtc_configuration={  # Add this config
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     }
#     # ...
# )
# Decode image
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Set background for local web
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)



# Process for solve_by_image page
def solve_by_image(upload_img, model):
    res_img = image_solver(upload_img, model)
    return res_img


############ Helper function for solve page by numbers##################

### CHECK VALID INPUT FROM USER

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

## DRAW INPUT AND RESULT 
def draw_gridline(target_shape = (600,600,3)):
    base_img = 1* np.ones(target_shape)
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


def draw_user_image(base_img, input_str):
    """
    Create sudoku quiz image from user input
    """
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



########################################################


# Process for real-time solver page
class realtime_solver(VideoProcessorBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        final_img = img.copy()
        to_process_img = img.copy()
        #Processing 
        thresholded_img = preprocess(to_process_img) # Gray-scale img
        corners_img, corners_list, org_img = find_contours(thresholded_img, to_process_img)

        if corners_list:
            # Warped original img 
            warped, matrix = warp_image(corners_list, corners_img)
            # Threshold warped img
            warped_processed = preprocess(warped) # warped_processed is gray-scaled img

            #Get lines
            horizontal = grid_line_helper(warped_processed, shape_location=0)
            vertical = grid_line_helper(warped_processed, shape_location=1)

            # Create mask
            grid_mask = create_grid_mask(horizontal, vertical)
            # Resize will get better result ??
            grid_mask = cv2.resize(grid_mask,(600,600), cv2.INTER_AREA)
            # Extract number
            number_img = cv2.bitwise_and(cv2.resize(warped_processed, (600,600), cv2.INTER_AREA), grid_mask)
            # number_img = cv2.bitwise_and(warped_processed, grid_mask)
            # Split into squares
            squares = split_squares(number_img)
            cleaned_squares = clean_square_all_images(squares)

            # Resize and scale pixel
            resized_list = resize_square(cleaned_squares)
            norm_resized = normalize(resized_list)

            # # Recognize digits
            rec_str = recognize_digits(classifier, norm_resized)
            board = convert_str_to_board(rec_str)
            
            # Solve
            unsolved_board = board.copy()
            sudoku = Sudoku_solver(board, 9)
            start_time = time.time()
            sudoku.solve()
            solved_board = sudoku.board
            # Unwarp
            _, warp_with_nums = draw_digits_on_warped(warped, solved_board, unsolved_board)

            final_img = unwarp_image(warp_with_nums, corners_img, corners_list, time.time() - start_time)
            return final_img
        
        return img

def home_page():
    """
    Home page content
    """
    html_render_1 = """</br>
                            </br>
                            <div style="background-color:#FFF5EE;
                            padding:8px; 
                            border: 5px dashed purple;
                            border-radius: 25px;
                            position: relative;
                            top: 10px; ">
                            <h4 style="color:black;text-align:center; font-family: cursive">
                            A simple sudoku solver app using OpenCV and CNN</h4>
                            </div>
                            </br>"""
    st.markdown(html_render_1, unsafe_allow_html=True)
    html_render_2 = """
    </br>
    </br>
    <div style="
                background-color:#c0ccc0;
                padding:10px; 
                border: 5px outset ;
                border-radius: 25px;
                position: relative;
                top: 30px;
                ">
                <h4 style="color:black; font-family: cursive">
                        This app has three functions:</h4>
                <ul>
                <li style ="color: black;font-family: cursive; ">Solve sudoku quiz from user's input numbers.</li>
                <li style ="color: black;font-family: cursive; ">Solve sudoku quiz from uploading image.</li>
                <li style ="color: black;font-family: cursive;" >Solve real-time sudoku quiz from Webcam.</li>
                </ul>
                </div>
                </br>"""
    st.markdown(html_render_2, unsafe_allow_html=True)
    st.text("")
    st.text("")
    sidebar_open = """
    <h5 style="color:black; font-family: cursive">Click the ">" arrow at the top-left corner to begin.</h5>
    """
    st.markdown(sidebar_open, unsafe_allow_html=True)
    home_image = Image.open("streamlit_app\Bg5.jpg")
    st.image(home_image)
    source_render = """
    <h5 style="color:black; font-family: cursive"> <a href = "https://github.com/LTPhat/Sudoku_Solver">Source Code</a></h5>
    """
    st.markdown(source_render, unsafe_allow_html=True)

def image_solve_page(model):

    header_image = """
        <div style="
                    background-color: #fff;
                    border-radius: 6px;
                    min-height: 80px;
                    --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                    box-shadow: var(--shadow);
                    display: flex;
                    margin: 0px;
                    padding: 0px;
                    border-radius: 25px;
                    box-sizing: border-box;
                    justify-content: center;
                    align-items: center;
                    color: transparent;
                    background-image: linear-gradient(115deg, #26940a, #162b10);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive; font-weight: bold">
                            Sudoku solver by image</h3>
                    </div>
                    </br>"""
    st.markdown(header_image, unsafe_allow_html=True)
    st.text("")
    st.text("")
    some_sample = """
    <h5 style="color:black; font-family: cursive"> Some sample result images:</h5>
    """
    st.markdown(some_sample, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("result_image_solver\pic1.png",width=230,caption= "Sample 1")
    with col2:
        st.image("result_image_solver\pic2.png",width=230,caption= "Sample 2")
    with col3:
        st.image("result_image_solver\pic3.png",width=230,caption= "Sample 3")
    recommend = """
    <h5 style="color:black; font-family: cursive"> Click the button </b><i><u>Browser file</u></i></b> to upload your image (png, jpeg, webp).</h5>
    """
    st.markdown(recommend, unsafe_allow_html=True)
    note = """
    <div style="
                background-color: #fff;
                min-height: 60px;
                --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                box-shadow: var(--shadow);
                border-radius: 15px 50px 30px 5px;
                box-sizing: border-box;
                justify-content: center;
                align-items: center;
                margin: 20px;
                padding: 20px;
                color: transparent;
                background-image: linear-gradient(115deg, #240c05,#0d0502);
                ">
                <h4 style="color:#fcc200; font-family: cursive; font-weight: bold; text-align: center;">
                        <u>Note:</u> Image with big size are recommended to get more accurate result.</h4>
                </div>
                </br>"""
    st.markdown(note, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose image file", accept_multiple_files=False)
    if uploaded_file is not None:
        st.write("File uploaded:", uploaded_file.name)
        show_img = Image.open(uploaded_file)
        show_img = np.array(show_img)
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        st.image(show_img, caption= "Original image uploaded")
        if not os.path.exists("streamlit_app\image_from_user"):
            os.mkdir("streamlit_app\image_from_user")
        save_dir = "streamlit_app\image_from_user"
        with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    if st.button("Solve"):
        saveimg_dir = "streamlit_app\image_from_user" + "\{}".format(uploaded_file.name)
        #Solve
        res_img = image_solver(saveimg_dir, model)
        st.image(res_img, caption="Result", clamp=True, channels='BGR')



def real_time_page(model):

    header_realtime = """
        <div style="
                    background-color: #fff;
                    border-radius: 6px;
                    min-height: 80px;
                    --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                    box-shadow: var(--shadow);
                    display: flex;
                    margin: 0px;
                    padding: 0px;
                    border-radius: 25px;
                    box-sizing: border-box;
                    justify-content: center;
                    align-items: center;
                    color: transparent;
                    background-image: linear-gradient(45deg, #1193d9, #1b2f69);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive">
                            Real-time sudoku solver</h3>
                    </div>
                    </br>"""
    st.markdown(header_realtime, unsafe_allow_html=True)
    some_sample ="""
    <h5 style="color:black; font-family: cursive">Some realtime result images:</h5>
    """
    st.markdown(some_sample, unsafe_allow_html=True)
    col1, col2= st.columns(2)
    with col1:
        st.image("real_time_img\Res4.png",width=400,caption= "Sample 1")
    with col2:
        st.image("real_time_img\Res5.png",width=400,caption= "Sample 2")

    col3, col4= st.columns(2)
    with col3:
        st.image("real_time_img\Res8.png",width=400,caption= "Sample 3")
    with col4:
        st.image("real_time_img\Res7_fail.png",width=400,caption= "Sample 4")

    webcam_here ="""
    <h5 style="color:black; font-family: cursive">Your Webcam here:</h5>
    """
    st.markdown(webcam_here, unsafe_allow_html=True)
    webrtc_streamer(key="key")

def about_sudoku_page():

    header_sudoku = """
        <div style="
                    background-color: #fff;
                    border-radius: 6px;
                    min-height: 80px;
                    --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                    box-shadow: var(--shadow);
                    display: flex;
                    margin: 0px;
                    padding: 0px;
                    border-radius: 25px;
                    box-sizing: border-box;
                    justify-content: center;
                    align-items: center;
                    color: transparent;
                    background-image: linear-gradient(115deg, #7a0421, #2e050f);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive">
                            What is sudoku?</h3>
                    </div>
                    </br>"""
    st.markdown(header_sudoku, unsafe_allow_html=True)
    content_render = """</br>
                        <div style="background-color:#FFF5EE;
                        padding:20px; 
                        border: 5px groove;
                        border-radius: 15px 50px 30px;
                        position: relative;
                        background-image: linear-gradient(115deg, #f5cfab, #a88d72);
                        top: 10px; ">
                        <p style = "color: black; font-size: 18px; font-family: cursive; text-align:justify;"><b>Sudoku</b> 
                        (originally called <b>Number Place</b>), is a logic-based, combinatorial number-placement puzzle. 
                        </p>
                        <p style = "color: black; font-sizeap: 18px; font-family: cursive; text-align:justify;">In classic Sudoku, the objective is to fill a 9 × 9 grid with digits so that each column, each row, and each of the nine 3 × 3 subgrids that compose the grid (also called "boxes", "blocks", or "regions") contain all of the digits from 1 to 9. 
                        The puzzle setter provides a partially completed grid, which for a well-posed puzzle has a single solution.</p>
                        </div>
                        </br>"""
    st.markdown(content_render, unsafe_allow_html=True)
    st.image("testimg\Real_test4.jpg",width=700 ,caption= "A typical sudoku puzzle")
    st.text("")
    st.text("")
    rule_render = """</br>
                        <div style="background-color:#FFF5EE;
                        padding:10px; 
                        min-height: 80px;
                        border-radius: 25px;
                        position: relative;
                        background-image: linear-gradient(115deg, #eb0e9a, #380726;
                        top: 10px; ">
                        <h3 style="color:#fcc200; font-family: cursive; text-align: center;">
                            Sudoku rules</h3>
                        </div>
                        </br>"""
    st.markdown(rule_render, unsafe_allow_html=True)
    rule_content = """<div style="background-color:#FFF5EE;
                        padding:40px;
                        border: 5px groove;
                        border-radius: 15px 50px 30px;
                        position: relative;
                        background-image: linear-gradient(115deg, #ede6eb, #a39da1);
                        top: 10px; ">
                        <p style = "color: blue; font-size: 20px; font-family: cursive; font-weight: bold;text-align:justify;"> Rule 1 - Each row must contain the numbers from 1 to 9, without repetitions.</p>
                        <p style = "color: black; font-size: 18px; font-family: cursive;text-align:justify;">The player must focus on filling each row of the grid while ensuring there are no duplicated numbers. The placement order of the digits is irrelevant.</p>
                        <p style = "color: black; font-size: 18px; font-family: cursive; text-align:justify;">Every puzzle, regardless of the difficulty level, begins with allocated numbers on the grid. The player should use these numbers as clues to find which digits are missing in each row.</p>
                        <p style = "color: blue; font-size: 20px; font-family: cursive; font-weight: bold;text-align:justify;"> Rule 2 - Each column must contain the numbers from 1 to 9, without repetitions.</p>
                        <p style = "color: black; font-size: 18px; font-family: cursive; text-align:justify;">The Sudoku rules for the columns on the grid are exactly the same as for the rows. The player must also fill these with the numbers from 1 to 9, making sure each digit occurs only once per column.</p>
                        <p style = "color: black; font-size: 18px; font-family: cursive; text-align:justify;">Every puzzle, regardless of the difficulty level, begins with allocated numbers on the grid. The numbers allocated at the beginning of the puzzle work as clues to find which digits are missing in each column and their position.</p>
                        <p style = "color: blue; font-size: 20px; font-family: cursive; font-weight: bold;text-align:justify;"> Rule 3 - The digits can only occur once per block (nonet).</p>
                        <p style = "color: black; font-size: 18px; font-family: cursive; text-align:justify;">A regular 9 x 9 grid is divided into 9 smaller blocks of 3 x 3, also known as nonets. The numbers from 1 to 9 can only occur once per nonet.</p>
                        <p style = "color: black; font-size: 18px; font-family: cursive; text-align:justify;">In practice, this means that the process of filling the rows and columns without duplicated digits finds inside each block another restriction to the numbers’ positioning.</p>
                        <p style = "color: blue; font-size: 20px; font-family: cursive; font-weight: bold;text-align:justify;"> Rule 4 - The sum of every single row, column and nonet must equal 45</p>
                        <p style = "color: black; font-size: 18px; font-family: cursive; text-align:justify;">To find out which numbers are missing from each row, column or block or if there are any duplicates, the player can simply count or flex their math skills and sum the numbers. When the digits occur only once, the total of each row, column and group must be of 45.</p>
                        </div>
                        """
    st.markdown(rule_content, unsafe_allow_html=True)
    st.text("")
    st.text("")
    how_to_play = """
    <div style="
                background-color: #fff;
                border-radius: 6px;
                min-height: 80px;
                --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                box-shadow: var(--shadow);
                display: flex;
                margin: 0px;
                padding: 0px;
                border-radius: 25px;
                box-sizing: border-box;
                justify-content: center;
                align-items: center;
                color: transparent;
                background-image: linear-gradient(115deg, #750bb8, #190426);
                ">
                <h3 style="color:#fcc200; font-family: cursive">
                        How to solve Sudoku puzzle</h3>
                </div>
                </br>"""
    st.markdown(how_to_play, unsafe_allow_html=True)
    st.video("https://youtu.be/kvU9_MVAiE0")
    st.text("")
    st.text("")
    back_track = """
    <div style="
                background-color: #fff;
                border-radius: 6px;
                min-height: 80px;
                --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                box-shadow: var(--shadow);
                display: flex;
                margin: 10px;
                padding: 10px;
                border-radius: 25px;
                box-sizing: border-box;
                justify-content: center;
                align-items: center;
                color: transparent;
                background-image: linear-gradient(115deg, #8f0909, #260303);
                ">
                <h3 style="color:#fcc200; font-family: cursive">
                Solve Sudoku by Backtracking Algorithm</h3>
                </div>
                </br>"""
    st.markdown(back_track, unsafe_allow_html=True)
    st.video("https://youtu.be/eqUwSA0xI-s")
    st.text("")
    st.video("https://youtu.be/lK4N8E6uNr4")

def number_solve_page():
    header_text = """
        <div style="
                    background-color: #fff;
                    border-radius: 6px;
                    min-height: 80px;
                    --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                    box-shadow: var(--shadow);
                    display: flex;
                    margin: 0px;
                    padding: 0px;
                    border-radius: 25px;
                    box-sizing: border-box;
                    justify-content: center;
                    align-items: center;
                    color: transparent;
                    background-image: linear-gradient(45deg, #803211, #381303);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive">
                            Sudoku solver by user's input numbers</h3>
                    </div>
                    </br>"""
    st.markdown(header_text, unsafe_allow_html=True)
    request ="""
    <h5 style="color:black; font-family: cursive">Fill in the sudoku quiz below.</h5>
    """
    st.markdown(request, unsafe_allow_html=True)
    note_term = """
    <div style="
                background-color: #fff;
                border-radius: 6px;
                min-height: 60px;
                --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                box-shadow: var(--shadow);
                display: flex;
                margin: 0px;
                padding: 0px;
                border-radius: 25px;
                box-sizing: border-box;
                justify-content: center;
                align-items: center;
                color: transparent;
                background-image: linear-gradient(45deg, #18224a, #090d1c);
                ">
                <h3 style="color:#fcc200; font-family: cursive">
                        Fill in the blank squares with 0!</h3>
                </div>
                </br>"""
    st.markdown(note_term, unsafe_allow_html= True)
    example_image = """
    <h5 style="color:black; font-family: cursive">Example image:</h5>
    """
    st.markdown(example_image, unsafe_allow_html=True)
    st.image("testimg\Real_test4.jpg", caption="Example image")
    example_input = """
    <h5 style="color:black; font-family: cursive">Example input:</h5>
    """
    st.markdown(example_input, unsafe_allow_html=True)
    st.text("")
    st.text("0 0 0 0 0 0 0 0 0")
    st.text("0 0 8 2 3 6 4 0 0")
    st.text("0 1 0 0 5 0 0 2 0")
    st.text("5 0 0 0 0 0 0 0 9")
    st.text("1 0 0 0 0 0 0 0 7")
    st.text("0 8 0 0 0 0 0 5 0")
    st.text("0 0 5 0 0 0 2 0 0")
    st.text("0 0 0 8 0 7 0 0 0")
    st.text("0 0 0 0 2 0 0 0 0")
    st.text("")
    your_input = """
    <h5 style="color:black; font-family: cursive">Your sudoku puzzle here:</h5>
    """
    st.markdown(your_input, unsafe_allow_html= True)

    input_str = ""
    with st.form(key='myform', clear_on_submit=False):
        for i in range(1,10):
            cols = st.columns(9)
            for j, col in enumerate(cols):
                cell = col.text_input("A[{}][{}]".format(i, j+1))
                input_str += cell
        submitButton = st.form_submit_button(label = 'Solve')
    if submitButton:
        if len(input_str) != 81:
            st.warning("Invalid input. Please check again")
            st.warning("Length of input must be 81 and space is not allow in following input string")
            st.success("Length of input string: {}".format(len(input_str)))
            st.error(input_str)
        elif valid_input_str(input_str) == False:
            base_img = draw_gridline(target_shape=(600,600,3))
            show_image_input = """
            <h5 style="color:black; font-family: cursive">Your sudoku puzzle:</h5>
            """
            st.markdown(show_image_input, unsafe_allow_html= True)
            in_image, board = draw_user_image(base_img, input_str)
            st.image(in_image, caption = "Input puzzle")
            st.error("Invalid puzzle (repetition on row, column or box). Please check again.")
        else:
            base_img = draw_gridline(target_shape=(600,600,3))
            show_image_input = """
            <h5 style="color:black; font-family: cursive">Your sudoku puzzle:</h5>
            """
            st.markdown(show_image_input, unsafe_allow_html= True)
            in_image, board = draw_user_image(base_img, input_str)
            st.image(in_image, caption = "Input puzzle")
            res_board, unsolve_board = solve(board)
            res_img = draw_result(base_img, unsolve_board, res_board)
            show_image_output = """
            <h5 style="color:black; font-family: cursive">Your result puzzle: </h5>
            """
            st.markdown(show_image_output, unsafe_allow_html= True)
            st.image(res_img, caption= "Result puzzle", clamp=True, channels='BGR')

def about_me_page():
    header_aboutme = """
        <div style="
                    background-color: #fff;
                    border-radius: 6px;
                    min-height: 80px;
                    --shadow: 1px 1px 1px 1px rgb(0 0 0 / 0.25);
                    box-shadow: var(--shadow);
                    display: flex;
                    margin: 0px;
                    padding: 0px;
                    border-radius: 25px;
                    box-sizing: border-box;
                    justify-content: center;
                    align-items: center;
                    color: transparent;
                    background-image: linear-gradient(115deg, #260636, #08030a);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive">
                            About me?</h3>
                    </div>
                    </br>"""
    st.markdown(header_aboutme, unsafe_allow_html=True)
    
    info = """
    </br>
    </br>
    <div style="background-color:#FFF5EE;
                        padding:20px; 
                        border: 5px groove;
                        border-radius: 15px 50px 30px;
                        position: relative;
                        background-image: linear-gradient(115deg, #c7c0bb,#4d4946);
                        top: 10px">
        <ul>
        <li style ="color: black;font-family: cursive; font-weight: bold">Hello, I'm Lam Thanh Phat.</li>
        <li style ="color: black;font-family: cursive; font-weight: bold">I'm a second-year student at Ho Chi Minh University of Technology (HCMUT).</li>
        <li style ="color: black;font-family: cursive; font-weight: bold">My Github:  <a href="https://github.com/LTPhat">My github</a></li>
        <li style ="color: black;font-family: cursive; font-weight: bold">My Facebook:  <a href="https://www.facebook.com/Nicedreamss">My fb</a></li>
        </ul>
    </div>
    </br>"""
    st.markdown(info, unsafe_allow_html=True)
def main(model):
    # Set background
    set_background("streamlit_app\Bg4.jpg")
    # Header and Sidebar
    st.markdown("<h1 style='text-align: center; color: #770737; font-family: cursive; padding: 40px; margin: 20px; font-size: 38px; '>Welcome to Sudoku Solver App</h1>",
                unsafe_allow_html=True)
    activities = ["Home", "Sudoku solver by number inputs", "Sudoku solver by image" ,"Real-time sudoku solver", "About sudoku" , "About me"]
    choice = st.sidebar.selectbox("Select your choice", activities)
    st.sidebar.markdown(
    """ Developed by Phat, HCMUT""")

    #Process Home Page
    if choice == "Home":
        home_page()

    # Process solve by image
    if choice == "Sudoku solver by image":
        image_solve_page(classifier)
    
    #Process real-time
    if choice == "Real-time sudoku solver":
        real_time_page(classifier)

    if choice == "About sudoku":
        about_sudoku_page()
        
    if choice == "Sudoku solver by number inputs":
        number_solve_page()
    if choice == "About me":
        about_me_page()

if __name__ == "__main__":
    main(classifier)