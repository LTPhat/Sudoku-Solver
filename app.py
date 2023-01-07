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


set_background("streamlit_app\Bg4.jpg")

# Process for solve_by_image page
def solve_by_image(upload_img, model):
    res_img = image_solver(upload_img, model)
    return res_img

# Draw sudoku grid with number from string

def draw_grid(target_shape = (600,600)):
    base_img = np.zeros(target_shape)
    return base_img

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


def main(model):

    # Header and Sidebar
    st.markdown("<h1 style='text-align:center; color: #770737; font-weight: bold; font-family: cursive; padding: 30px '>Welcome to Sudoku Solver </h1>",
                unsafe_allow_html=True)
    activities = ["Home", "Sudoku solver by image", "Sudoku solver by number inputs","About sudoku" ,"Real-time sudoku solver", "Overall process to solve", "About me"]
    choice = st.sidebar.selectbox("Select your choice", activities)
    st.sidebar.markdown(
    """ Developed by Phat, HCMUT""")



    #Process Home Page
    if choice == "Home":
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
                            This app has two functions:</h4>
                    <ul>
                    <li style ="color: black;font-family: cursive; ">Solve sudoku quiz through uploading image.</li>
                    <li style ="color: black;font-family: cursive;" >Solve real-time sudoku quiz through camera.</li>
                    </ul>
                    </div>
                    </br>"""
        st.markdown(html_render_2, unsafe_allow_html=True)
        st.text("")
        st.text("")
        home_image = Image.open("streamlit_app\Bg5.jpg")
        st.image(home_image)




    # Process solve by image
    if choice == "Sudoku solver by image":
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
            st.image("result_image_solver\pic1.png",width=250,caption= "Sample 1")
        with col2:
            st.image("result_image_solver\pic2.png",width=250,caption= "Sample 2")
        with col3:
            st.image("result_image_solver\pic3.png",width=250,caption= "Sample 3")

        recommend = """
        <h5 style="color:black; font-family: cursive"> Click the button </b><i><u>Browser file</u></i></b> to upload your image.</h5>
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
                    padding: 10px;
                    color: transparent;
                    background-image: linear-gradient(115deg, #240c05,#0d0502);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive; font-weight: bold; text-align: justify;">
                            <u>Note:</u> Image with big size are recommended.</h3>
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
            st.image(res_img, caption="Result")
    



    #Process real-time
    if choice == "Real-time sudoku solver":
        
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
        webrtc_streamer(key="key")





    if choice == "About sudoku":
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
                            <p style = "color: black; font-size: 18px; font-family: cursive"; text-align:justify;><b>Sudoku</b> 
                            (originally called <b>Number Place</b>), is a logic-based, combinatorial number-placement puzzle. 
                            </p>
                            <p style = "color: black; font-sizeap: 18px; font-family: cursive"; text-align:justify;>In classic Sudoku, the objective is to fill a 9 × 9 grid with digits so that each column, each row, and each of the nine 3 × 3 subgrids that compose the grid (also called "boxes", "blocks", or "regions") contain all of the digits from 1 to 9. 
                            The puzzle setter provides a partially completed grid, which for a well-posed puzzle has a single solution.</p>
                            </div>
                            </br>"""
        st.markdown(content_render, unsafe_allow_html=True)
        st.text("")
        st.text("")
        st.image("testimg\Real_test4.jpg",width=700 ,caption= "A typical sudoku puzzle")
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
    




    if choice == "Sudoku solver by number inputs":
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
                            Sudoku solver by text input</h3>
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
        <h5 style="color:black; font-family: cursive">Example image</h5>
        """
        st.markdown(example_image, unsafe_allow_html=True)
        st.image("testimg\Real_test4.jpg", caption="Example image:")
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
        <h5 style="color:black; font-family: cursive">Your sudoku quiz:</h5>
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
                st.success(len(input_str))
                st.error(input_str)
            else:
                base_img = draw_grid(target_shape=(600,600))
                board = convert_str_to_board(input_str)
                sudoku = Sudoku_solver(board, size = 9)
                sudoku.solve()
                res_board = sudoku.board
                result_img = draw_digits_on_warped(base_img, res_board, board)
                st.image(result_img, caption= "Result image")
if __name__ == "__main__":
    main(classifier)