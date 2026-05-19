# Title
Basic Image Processing Operations using OpenCV in Python

# Objective :
To perform basic image processing operations using OpenCV in Python such as:

1. Reading and displaying an image  
2. Converting a color image to grayscale image  
3. Converting an image into numerical arrays  
4. Rotating an image  
5. Converting RGB color model to CMY color model  

# Software Requirements

- Python
- Google Colab
- OpenCV Library
- NumPy Library
- imutils Library

# Theory

Image processing is a technique used to manipulate digital images using computer algorithms. It is widely used in computer vision, medical imaging, pattern recognition, and artificial intelligence applications.

OpenCV (Open Source Computer Vision Library) is an open-source library used for image processing and computer vision tasks. It provides various functions to read, display, modify, and analyze images.

A grayscale image contains only intensity information and reduces image complexity. Numerical arrays represent image pixels in matrix form for computation and analysis. Image rotation changes the orientation of the image by a specified angle. RGB is an additive color model consisting of Red, Green, and Blue components, whereas CMY is a subtractive color model consisting of Cyan, Magenta, and Yellow components.

# Program 1: Read and Display Image

```python
import cv2

image = cv2.imread('/content/download.jpg')

from google.colab.patches import cv2_imshow

cv2_imshow(image)

print('Original Image')
```


# Program 2: Convert a Color Image to Grayscale Image

```python
import cv2

image1 = cv2.imread('/content/download.jpg', 0)

from google.colab.patches import cv2_imshow

cv2_imshow(image1)

print('Grayscale image')
```


# Program 3: Convert an Image into Number Arrays


```python
import cv2

image = cv2.imread('/content/download.jpg')

from google.colab.patches import cv2_imshow

cv2_imshow(image)

from numpy import asarray

data = asarray(image)

print(data)
```


# Program 4: Image Rotation

```python
import cv2
import imutils

image = cv2.imread('/content/download.jpg')

from google.colab.patches import cv2_imshow

cv2_imshow(image)

print('Original Image')

rotated = imutils.rotate(image, 90)

cv2_imshow(rotated)

print('90 degree rotated image')
```


# Program 5: Conversion of RGB Model to CMY Model

```python
import cv2

image = cv2.imread('/content/download.jpg')

from google.colab.patches import cv2_imshow

cv2_imshow(image)

print('Original Image')

R, G, B = cv2.split(image)

C = 255 - R
M = 255 - G
Y = 255 - B

cv2_imshow(R)
print('Red Component')

cv2_imshow(G)
print('Green Component')

cv2_imshow(B)
print('Blue Component')

cv2_imshow(C)
print('Cyan Component')

cv2_imshow(M)
print('Magenta Component')

cv2_imshow(Y)
print('Yellow Component')
```

# Conclusion

In this experiment, various image processing operations were performed using OpenCV in Python. The image was successfully displayed, converted into grayscale format, transformed into numerical arrays, rotated by 90 degrees, and converted from RGB color model to CMY color model. The objectives of the experiment were achieved successfully.