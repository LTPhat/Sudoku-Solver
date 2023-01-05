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


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(
    # ...
    rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
    # ...
)
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
    activities = ["Home", "Sudoku solver by image", "Real-time sudoku solver" , "Overall process to solve", "About"]
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
                    background-image: linear-gradient(115deg, #7f11d9, #2c1a3b);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive; font-weight: bold">
                            Sudoku solver by image</h3>
                    </div>
                    </br>"""
        st.markdown(header_image, unsafe_allow_html=True)
        st.text("")
        st.text("")
        some_sample = """
        <h4 style="color:black; font-family: cursive"> Some sample result images:</h4>
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
        <h6 style="color:black; font-family: cursive"> Click the button to upload your image. Images with big shape are recommended.</h6>
        """
        st.markdown(recommend, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose image file", accept_multiple_files=False)
        if uploaded_file is not None:
            st.write("File uploaded:", uploaded_file.name)
            show_img = Image.open(uploaded_file)
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
                    background-image: linear-gradient(115deg, #1193d9, #1b2f69);
                    ">
                    <h3 style="color:#fcc200; font-family: cursive">
                            Real-time sudoku solver</h3>
                    </div>
                    </br>"""
        st.markdown(header_realtime, unsafe_allow_html=True)
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=realtime_solver)


if __name__ == "__main__":
    main(classifier)