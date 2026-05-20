# Title : 
Image Transformations

# Objective
* To detect edges in an image using the Canny Edge Detection method.
* To perform image scaling (resizing) using OpenCV functions.
* To apply different types of thresholding techniques on grayscale images.
* To perform logarithmic transformation for enhancing low-intensity pixel values in an image.
* To apply gamma transformation for brightness correction and contrast enhancement.
* To understand the practical use of image transformation techniques in digital image processing and computer vision applications.


# Theory
* ***Scaling :*** Scaling is just resizing of the image. OpenCV comes with a function `cv2.resize()` for
this purpose. The size of the image can be specified manually, or you can specify the scaling
factor. 
* ***Thresholding :*** Here, the matter is straight forward. If pixel value is greater than a threshold
value, it is assigned one value (may be white), else it is assigned another value (may be black).
The function used is `cv2.threshold`. First argument is the source image, which should be a
grayscale image. Second argument is the threshold value which is used to classify the pixel
values. Third argument is the maxVal which represents the value to be given if pixel value is
more than (sometimes less than) the threshold value. OpenCV provides different styles of
thresholding and it is decided by the fourth parameter of the function.

# Software Requirements
- Python
- Google Colab
- OpenCV Library
- NumPy Library
- imutils Library

# Lab Work 
1. Progam to Detect edge of an image
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/content/Einstein.jpg',0)
edges = cv2.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
```
2. Write a program to scale an image
```python 
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/content/j.png')
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.grid(True)
plt.subplot(122),plt.imshow(res),plt.title('Scaling')
plt.grid(True)
plt.show()

```
<
3. Write a program to thresholding on an image
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("/content/Einstein.jpg",0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
titles = ["Original Image","BINARY","BINARY_INV","TRUNC","TOZERO","TOZERO_INV"]
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
  plt.subplot(2,3,i+1),plt.imshow(images[i],"gray")
  plt.title(titles[i])
  plt.xticks([]),plt.yticks([])
plt.show()
```

4. Log Transformation
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/content/Einstein.jpg', 0)
c = 255 / np.log(1 + np.max(img))
log_transformed = c * (np.log(img + 1))
log_transformed = np.array(log_transformed, dtype=np.uint8)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(log_transformed, cmap='gray')
plt.title('Log Transformed Image')
plt.xticks([]), plt.yticks([])
plt.show()
```

5. Gama Transformation
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('/content/Einstein.jpg', 0)
gamma = 2.0    # The value of Gama increases the image darkness increases
gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gamma_corrected, cmap='gray')
plt.title('Gamma Transformed Image')
plt.xticks([]), plt.yticks([])
plt.show()
```

# Conclusion
In this lab, various image transformation techniques were successfully performed using OpenCV in Python. Edge detection was carried out using the Canny Edge Detection method to identify object boundaries in images. Image scaling was implemented to resize images using different interpolation methods. Different types of thresholding techniques were applied to convert grayscale images into binary and segmented forms. Log transformation was performed to enhance darker regions of images, while gamma transformation was used to adjust image brightness and contrast. Through these experiments, the practical applications of image transformation techniques in digital image processing and computer vision were understood successfully.