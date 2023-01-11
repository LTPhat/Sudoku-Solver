# Sudoku Solver

- Sudoku solver app using ``OpenCV`` for image processing and ``Pytorch`` for training model on customed printed digits dataset.

- Final project for ``CourseWork 2`` in AI4E course.

- `Finished time:` 12/01/2023.

## About the dataset and model

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
![](https://github.com/LTPhat/Sudoku_Solver/blob/main/processing_image/sample2/complete_grid_line_using_houghLine.png)  
|[[6 9 3 5 0 0 0 0 0]
  [0 0 0 3 0 0 2 5 0]
  [0 0 0 0 0 4 0 3 8]
  [0 0 0 0 7 6 4 0 3]
  [1 0 0 0 0 0 0 0 2]
  [8 0 3 2 8 0 0 0 0]
  [4 5 0 1 0 0 0 0 0]
  [0 8 6 0 0 5 0 0 0]
  [0 7 0 0 8 0 0 0 0]]


## Sudoku Solver App

- ``Streamlit`` is a free and open-source framework to rapidly build and share beautiful machine learning and data science web apps. It is a Python-based library specifically designed for machine learning engineers.

- To run app, type the following line on cmd:
  ```
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
