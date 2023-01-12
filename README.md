# Sudoku Solver

- Sudoku solver app using ``OpenCV`` for image processing and ``Pytorch`` for training model on customed printed digits dataset.

- `Finished time:` 12/01/2023.

## About dataset and model

- We don't use MNIST dataset because:

  +) A large number of digits from input images are printed digits (digital digits).
 
   +) There are some samples which have wrong labels (noise), the model for sudoku digits classifier is need to be as accurate as possible.
 
   +) There are some mistakes recognizing digits (especially 0 and 1, 3 and 8) when using MNIST dataset.
  
 - In this repo, we create a custom printed digit dataset by drawing digits with various popular font styles (Times New Roman, Tahoma, Arial, Hevectica ...) which are shown in ``font`` folder.
 
 ### Loading custom dataset in Pytorch
 
- A custom Dataset class must implement three functions: ``__init__``, ``__len__``, and ``__getitem__``. 

- Tutorial link: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html.

- ``PrintedMNIST`` class for custom dataset with Pytorch:

```python
 class PrintedMNIST(Dataset):
    """
    Generate digital mnist dataset for digits recognition
    """
    def __init__(self, samples, random_state, transform = None):
        self.samples = samples
        self.random_state = random_state
        self.transfrom = transform
        self.fonts = fonts
        random.seed(random_state)
        
    def __len__(self):
        return self.samples
        
    def __getitem__(self, index):
        color = random.randint(200,255)
        
        #Generate image
        img = Image.new("L",(256, 256))
        label = random.randint(0,9)
        size = random.randint(180, 220)
        x = random.randint(60, 80)
        y = random.randint(30, 60)
        draw = ImageDraw.Draw(img)
        
        #Choose random font style in font style list
        font = ImageFont.truetype(random.choice(self.fonts), size)
        draw.text((x,y), str(label), color, font = fonts)
        img = img.resize((28,28), Image.BILINEAR)
        if self.transfrom:
            img = self.transfrom(img)
        return img, label
```
- ``Model``: Pretrained `ResNet50`, adjust the first convolutional layer to feed gray-scale image after image processing step.


## Overall Image Processing Pipeline

Original Image          |  Adaptive Threshold
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/original.png)  |![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/threshold.png)


Find corners          |  Perspective Transform
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/find_coner.png)  |![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/original_warped.png)


Find grid mask         |  Extract digit images
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/complete_grid_line_using_houghLine.png)  |![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/split_81_squares.png)


Clean noises        |   Recognize digits
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/cleaned_squares.png)  |![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/rec_digit.png)


Solve board       |  Draw solved board on image
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/solve.png)  |![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/draw_digits_on_warped.png)


Inverse Perspective Transform       |  Final result
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/final_result.png)  |![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/final_result.png)

## Note
  Change the url variable to your image url.
  
- Test sudoku solver by image on local, run ``image_solver.py``.

- Test real-time solver on local, run ``realtime_solver.py``.

- Test the entire image processing pipeline, run ``test_modules.py``.

## Sudoku Solver App

- ``Streamlit`` is a free and open-source framework to rapidly build and share beautiful machine learning and data science web apps. It is a Python-based library specifically designed for machine learning engineers.

- To run app, type the following line on cmd:
  ```python
  streamlit run app.py
  ```
- The Web application has 6 pages: 
   - ``Home page``
   - ``Sudoku solver by number inputs.``
   - ``Sudoku solver by image.``
   - ``Real-time sudoku solver.`` 
   - ``About Sudoku.``
   - ``About me.``
### Home Page
  
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/input_number_page/home_page.jpg)
  
### Sudoku solver by number inputs page
  
 - Surface.  
  
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/input_number_page/number_input_page.jpg)

 - Get input sudoku puzzle from user.
 
Empty puzzle           |  User input
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/input_number_page/your_sudoku_here.jpg)  |![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/input_number_page/input_puzzle.jpg)

- Generate input image and solved image from input puzzle.

Input image           |  Result image
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/input_number_page/your_puzzle.jpg)  |  ![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/input_number_page/your_result_puzzle.jpg)


### Sudoku solver by image inputs page

- Surface.  
  
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/image_page/surface.jpg)

- Upload image and get result image.

Uploaded image           |  Result image
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/image_page/upload_image.jpg)  |  ![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/image_page/solved_image.jpg)


### Real-time sudoku solver page

- Surface.  
  
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/realtime_page/realtime.jpg)


### About Sudoku page

- Some information and rules of this game.

![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/about_sudoku_page/definition.jpg)


Rules          |  How to play
:-------------------------:|:-------------------------:
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/about_sudoku_page/rules.jpg)  |![]![](https://github.com/LTPhat/Sudoku_Solver/blob/main/display_images/about_sudoku_page/another_sudoku.jpg)


### About me page

- Just some personal information.

## References

[1] WRITING CUSTOM DATASETS, DATALOADERS AND TRANSFORMS, https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.

[2] Perspective Transfromation, https://theailearner.com/2020/11/06/perspective-transformation/.

[3] Basic concepts of the homography explained with code, https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html.

[4] Image Thresholding, https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

[5] Understanding Homography (a.k.a Perspective Transformation), https://towardsdatascience.com/understanding-homography-a-k-a-perspective-transformation-cacaed5ca17.

[6] Real-Time-Sudoku-Solver-OpenCV-and-Keras, https://github.com/snehitvaddi/Real-Time-Sudoku-Solver-OpenCV-and-Keras.

[7] Morphological Transformations, https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html. 

