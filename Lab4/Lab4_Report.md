# Python Programming: Digital Image Processing

| Field | Details |
|-------|---------|
| **Name** | Oshan |
| **Roll No** | 230328 |
| **Faculty** | BCE |
| **Lab** | Python Programming — Digital Image Processing |

---

## Title
**Image Processing — Noise Models, Denoising, and Morphological Operations**

---

## Objective

- To add different types of noise (Gaussian, Rayleigh, Erlang, Exponential, Uniform, Salt & Pepper) to an image and observe their effects.
- To apply denoising techniques (Gaussian Blur, Median Blur, Non-Local Means) to restore noisy images.
- To perform morphological operations — Erosion, Dilation, Opening, Closing, and Gradient — on images and observe structural changes.

---

## Theory

### Part I: Image Noise Models

Noise in digital images refers to random variation in pixel intensity, typically introduced during acquisition or transmission.

| Noise Type | Description |
|------------|-------------|
| Gaussian Noise | Additive noise following a normal distribution; models sensor/thermal noise |
| Rayleigh Noise | Follows Rayleigh distribution; common in range imaging |
| Erlang (Gamma) Noise | Follows Gamma distribution; models laser speckle noise |
| Exponential Noise | One-sided distribution; produces brighter-shifted noise |
| Uniform Noise | Uniform random values added; all intensities equally probable |
| Salt & Pepper Noise | Impulsive noise; random pixels set to max (salt) or min (pepper) |

---

### Part II: Image Denoising Techniques

| Method | Description |
|--------|-------------|
| Gaussian Blur | Convolves image with Gaussian kernel; smooths out noise |
| Median Blur | Replaces pixel with the median of neighbors; best for salt & pepper |
| Non-Local Means (NLM) | Averages similar patches across the image; preserves fine detail |

---

### Part III: Morphological Operations

Morphological operations process images based on their shape using a **structuring element (kernel)**.

| Operation | Function |
|-----------|----------|
| Erosion | Shrinks bright regions; removes small noise |
| Dilation | Expands bright regions; fills gaps |
| Opening | Erosion followed by Dilation; removes small bright spots |
| Closing | Dilation followed by Erosion; fills small dark holes |
| Gradient | Difference between Dilation and Erosion; highlights object boundaries |

---

## Lab Tasks

---

### Task 1: Adding Gaussian Noise to an Image (Basic)

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Read image
img = cv2.imread('/content/Lenna.jpg', 0) / 255
cv2_imshow(img * 255)
print("Original Image")

# Generate Gaussian noise
noise = np.random.normal(0, 0.1, img.shape)
cv2_imshow((noise * 255).clip(0, 255))
print("Gaussian Noise")

# Add noise to image
noisy_img = img + noise
cv2_imshow((noisy_img * 255).clip(0, 255))
print("Noisy Image")
```

#### Output

![Task 1 Output](task1_output.png)

#### Observation

The image was normalized to [0, 1] and Gaussian noise with mean 0 and standard deviation 0.1 was generated and added. The resulting image shows visible graininess distributed uniformly across the entire image, typical of Gaussian noise behavior.

---

### Task 2: Adding Gaussian Noise to an Image (Detailed)

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

%matplotlib inline

# 1. Load and prepare image
img = cv2.imread('/content/Lenna.jpg', 0) / 255.0

# 2. Display original image
print("=== Original Image ===")
cv2_imshow(img * 255)

# 3. Generate Gaussian noise
noise_intensity = 0.1
noise = np.random.normal(loc=0, scale=noise_intensity, size=img.shape)

# 4. Display the noise
print("\n=== Gaussian Noise ===")
cv2_imshow((noise * 255).clip(0, 255))

# 5. Add noise to image and clip values
noisy_img = np.clip(img + noise, 0, 1)

# 6. Display noisy image
print("\n=== Noisy Image ===")
cv2_imshow(noisy_img * 255)

# 7. Show all images together for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noise, cmap='gray')
plt.title(f'Gaussian Noise (σ={noise_intensity})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Task 2 Output](task2_output.png)

#### Observation

Gaussian noise with σ = 0.1 was added and the noisy image was clipped to [0, 1]. The side-by-side comparison clearly shows the original, the noise pattern (centered around zero), and the degraded noisy image with random intensity fluctuations spread across all regions.

---

### Task 3: Adding Rayleigh Noise to an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

%matplotlib inline

# 1. Load and prepare image
img = cv2.imread('/content/Lenna.jpg', 0) / 255.0

# 2. Display original image
print("=== Original Image ===")
cv2_imshow(img * 255)

# 3. Generate Rayleigh noise
scale = 0.1
rayleigh_noise = np.random.rayleigh(scale, size=img.shape)

# 4. Display the noise
print("\n=== Rayleigh Noise ===")
cv2_imshow((rayleigh_noise * 255).clip(0, 255))

# 5. Add noise to image and clip values
noisy_img = np.clip(img + rayleigh_noise, 0, 1)

# 6. Display noisy image
print("\n=== Noisy Image (Rayleigh) ===")
cv2_imshow(noisy_img * 255)

# 7. Show all images together for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(rayleigh_noise, cmap='gray')
plt.title(f'Rayleigh Noise (scale={scale})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image (Rayleigh)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Photo not placce or image problem](task3_output.png)

#### Observation

Rayleigh noise, unlike Gaussian, is always non-negative and right-skewed. This causes the image to appear overall brighter and more washed out since pixel intensities are predominantly increased. The noise pattern shows a characteristic asymmetric distribution.

---

### Task 4: Adding Erlang (Gamma) Noise to an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from scipy.stats import gamma

%matplotlib inline

# 1. Load and prepare image
img = cv2.imread('/content/Lenna.jpg', 0) / 255.0

# 2. Display original image
print("=== Original Image ===")
cv2_imshow(img * 255)

# 3. Generate Erlang (Gamma) noise
shape = 2.0   # k (shape parameter)
scale = 0.1   # θ (scale parameter)
erlang_noise = np.random.gamma(shape=shape, scale=scale, size=img.shape)

# 4. Display the noise
print("\n=== Erlang Noise ===")
cv2_imshow((erlang_noise * 255).clip(0, 255))

# 5. Add noise to image and clip values
noisy_img = np.clip(img + erlang_noise, 0, 1)

# 6. Display noisy image
print("\n=== Noisy Image (Erlang) ===")
cv2_imshow(noisy_img * 255)

# 7. Show all images together for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(erlang_noise, cmap='gray')
plt.title(f'Erlang Noise (k={shape}, θ={scale})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image (Erlang)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Task 4 Output](task4_output.png)

#### Observation

Erlang (Gamma) noise with shape k = 2 and scale θ = 0.1 is also non-negative and tends to brighten the image. The noise distribution is smoother than Rayleigh and produces a speckle-like degradation, commonly seen in laser imaging systems.

---

### Task 5: Adding Exponential Noise to an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from scipy.stats import expon

%matplotlib inline

# 1. Load and prepare image
img = cv2.imread('/content/Lenna.jpg', 0) / 255.0

# 2. Display original image
print("=== Original Image ===")
cv2_imshow(img * 255)

# 3. Generate Exponential noise
scale = 0.1
exponential_noise = np.random.exponential(scale=scale, size=img.shape)

# 4. Display the noise
print("\n=== Exponential Noise ===")
cv2_imshow((exponential_noise * 255).clip(0, 255))

# 5. Add noise to image and clip values
noisy_img = np.clip(img + exponential_noise, 0, 1)

# 6. Display noisy image
print("\n=== Noisy Image (Exponential) ===")
cv2_imshow(noisy_img * 255)

# 7. Show all images together for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(exponential_noise, cmap='gray')
plt.title(f'Exponential Noise (λ={1/scale:.1f})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image (Exponential)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Task 5 Output](task5_output.png)

#### Observation

Exponential noise (λ = 10) is always positive and heavily right-skewed, causing the image to appear significantly brighter. The noise produces streaky bright artifacts, as many values are near zero but occasional large values cause bright spikes in pixel intensity.

---

### Task 6: Adding Uniform Noise to an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from scipy.stats import uniform

%matplotlib inline

# 1. Load and prepare image
img = cv2.imread('/content/Lenna.jpg', 0) / 255.0

# 2. Display original image
print("=== Original Image ===")
cv2_imshow(img * 255)

# 3. Generate Uniform noise
low = -0.2
high = 0.2
uniform_noise = np.random.uniform(low=low, high=high, size=img.shape)

# 4. Display the noise
print("\n=== Uniform Noise ===")
cv2_imshow(((uniform_noise - low) * 255/(high-low)).clip(0, 255))

# 5. Add noise to image and clip values
noisy_img = np.clip(img + uniform_noise, 0, 1)

# 6. Display noisy image
print("\n=== Noisy Image (Uniform) ===")
cv2_imshow(noisy_img * 255)

# 7. Show all images together for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(uniform_noise, cmap='gray', vmin=low, vmax=high)
plt.title(f'Uniform Noise ({low} to {high})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image (Uniform)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Task 6 Output](task6_output.png)

#### Observation

Uniform noise in the range [−0.2, 0.2] adds equal probability perturbations to all pixels. The result shows a uniformly grainy texture without any particular bias toward brighter or darker regions, distinguishing it from the one-sided noise types above.

---

### Task 7: Adding Salt and Pepper Noise to an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

%matplotlib inline

# 1. Load and prepare image
img = cv2.imread('/content/Lenna.jpg', 0) / 255.0

# 2. Display original image
print("=== Original Image ===")
cv2_imshow(img * 255)

# 3. Generate Salt and Pepper noise
def add_salt_pepper(image, prob=0.05):
    output = np.copy(image)
    # Salt noise (1)
    salt = np.random.rand(*image.shape) < prob/2
    output[salt] = 1
    # Pepper noise (0)
    pepper = np.random.rand(*image.shape) < prob/2
    output[pepper] = 0
    return output

noise_prob = 0.05
noisy_img = add_salt_pepper(img, prob=noise_prob)

# 4. Create noise-only visualization
noise_display = np.zeros_like(img)
salt_pixels = (noisy_img == 1)
pepper_pixels = (noisy_img == 0)
noise_display[salt_pixels] = 1
noise_display[pepper_pixels] = -1

# 5. Display the noise
print("\n=== Salt and Pepper Noise ===")
cv2_imshow(((noise_display + 1) * 127.5).clip(0, 255))

# 6. Display noisy image
print("\n=== Noisy Image (Salt & Pepper) ===")
cv2_imshow(noisy_img * 255)

# 7. Show all images together for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noise_display, cmap='gray', vmin=-1, vmax=1)
plt.title(f'Salt & Pepper Noise (p={noise_prob})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image (Salt & Pepper)')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Task 7 Output](task7_output.png)

#### Observation

Salt and Pepper noise (probability = 0.05) randomly sets 2.5% of pixels to pure white (salt) and 2.5% to pure black (pepper). The noise is clearly visible as scattered isolated bright and dark spots across the image, which is distinctly different in character from continuous noise models.

---

### Task 8: Gaussian Noise Addition and Denoising (Gaussian Blur, Median Blur, Non-Local Means)

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

%matplotlib inline

# 1. Load and prepare image
img = cv2.imread('/content/Lenna.jpg', 0) / 255.0

# 2. Display original image
print("=== Original Image ===")
cv2_imshow(img * 255)

# 3. Generate Gaussian noise
noise_intensity = 0.1
noise = np.random.normal(loc=0, scale=noise_intensity, size=img.shape)

print("\n=== Gaussian Noise ===")
cv2_imshow((noise * 255).clip(0, 255))

# 4. Add noise to image
noisy_img = np.clip(img + noise, 0, 1)

print("\n=== Noisy Image ===")
cv2_imshow(noisy_img * 255)

# 5. Denoising Methods
# Gaussian Blur
gaussian_denoised = cv2.GaussianBlur(
    (noisy_img * 255).astype(np.uint8), (5, 5), 1.5) / 255.0

# Median Blur
median_denoised = cv2.medianBlur(
    (noisy_img * 255).astype(np.uint8), 5) / 255.0

# Non-Local Means Denoising
nlm_denoised = cv2.fastNlMeansDenoising(
    (noisy_img * 255).astype(np.uint8),
    None, h=10, templateWindowSize=7, searchWindowSize=21) / 255.0

print("\n=== Gaussian Blur Denoised ===")
cv2_imshow(gaussian_denoised * 255)

print("\n=== Median Blur Denoised ===")
cv2_imshow(median_denoised * 255)

print("\n=== Non-Local Means Denoised ===")
cv2_imshow(nlm_denoised * 255)

# 6. Show all images together for comparison
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(noise, cmap='gray')
plt.title(f'Gaussian Noise (σ={noise_intensity})')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(gaussian_denoised, cmap='gray', vmin=0, vmax=1)
plt.title('Gaussian Blur Denoised')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(median_denoised, cmap='gray', vmin=0, vmax=1)
plt.title('Median Blur Denoised')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(nlm_denoised, cmap='gray', vmin=0, vmax=1)
plt.title('Non-Local Means Denoised')
plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Task 8 Output](task8_output.png)

#### Observation

| Denoising Method | Observed Effect |
|-----------------|----------------|
| Gaussian Blur | Reduces Gaussian noise effectively but causes slight blurring of edges |
| Median Blur | Performs well on Gaussian noise while preserving edges better than Gaussian blur |
| Non-Local Means | Produces the cleanest result — removes noise while preserving fine texture details |

---

### Task 9: Adding Multiple Noise Types and Denoising All

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load Image
img = cv2.imread('/content/Lenna.jpg', 0)
img = img.astype(np.float32) / 255.0

# Noise Functions
def add_rayleigh_noise(image, scale=0.1):
    noise = np.random.rayleigh(scale, image.shape)
    return np.clip(image + noise, 0, 1)

def add_erlang_noise(image, shape=2, scale=0.05):
    noise = np.random.gamma(shape, scale, image.shape)
    return np.clip(image + noise, 0, 1)

def add_uniform_noise(image, low=-0.1, high=0.1):
    noise = np.random.uniform(low, high, image.shape)
    return np.clip(image + noise, 0, 1)

def add_exponential_noise(image, scale=0.1):
    noise = np.random.exponential(scale, image.shape)
    return np.clip(image + noise, 0, 1)

def add_salt_pepper_noise(image, prob=0.05):
    noisy = np.copy(image)
    salt = np.random.rand(*image.shape) < prob/2
    pepper = np.random.rand(*image.shape) < prob/2
    noisy[salt] = 1
    noisy[pepper] = 0
    return noisy

# Add Noises
rayleigh_img    = add_rayleigh_noise(img)
erlang_img      = add_erlang_noise(img)
uniform_img     = add_uniform_noise(img)
exponential_img = add_exponential_noise(img)
sp_img          = add_salt_pepper_noise(img)

# Denoising
rayleigh_denoised    = cv2.GaussianBlur(rayleigh_img, (5,5), 0)
erlang_denoised      = cv2.GaussianBlur(erlang_img, (5,5), 0)
uniform_denoised     = cv2.GaussianBlur(uniform_img, (5,5), 0)
exponential_denoised = cv2.GaussianBlur(exponential_img, (5,5), 0)

sp_denoised = cv2.medianBlur((sp_img*255).astype(np.uint8), 5)
sp_denoised = sp_denoised.astype(np.float32) / 255.0

# Display Results
titles = [
    "Original",
    "Rayleigh Noise", "Rayleigh Denoised",
    "Erlang Noise", "Erlang Denoised",
    "Uniform Noise", "Uniform Denoised",
    "Exponential Noise", "Exponential Denoised",
    "Salt & Pepper Noise", "Salt & Pepper Denoised"
]

images = [
    img,
    rayleigh_img, rayleigh_denoised,
    erlang_img, erlang_denoised,
    uniform_img, uniform_denoised,
    exponential_img, exponential_denoised,
    sp_img, sp_denoised
]

plt.figure(figsize=(16,10))

for i in range(len(images)):
    plt.subplot(4,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
```

#### Output

![Task 9 Output](task9_output.png)

#### Observation

| Noise Type | Denoising Method | Observation |
|------------|-----------------|-------------|
| Rayleigh | Gaussian Blur | Significantly reduces the one-sided bright noise |
| Erlang | Gaussian Blur | Smooths out the gamma-distributed speckles |
| Uniform | Gaussian Blur | Effectively removes symmetric uniform noise |
| Exponential | Gaussian Blur | Reduces bright spike noise; some residual remains |
| Salt & Pepper | Median Filter | Best method for impulsive noise; restores image cleanly |

---

### Task 10: Erosion of an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/Lenna.jpg', 0)
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(erosion, cmap='gray'), plt.title("Erosion")
plt.xticks([]), plt.yticks([])
plt.show()
```

#### Output

![Task 10 Output](task10_output.png)

#### Observation

Erosion with a 3×3 kernel shrinks bright (foreground) regions by replacing each pixel with the minimum value in its neighborhood. This removes thin bright structures and small noise, and slightly darkens the overall image. Boundaries of bright objects are eroded inward.

---

### Task 11: Dilation of an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/Lenna.jpg', 0)
kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(img, kernel, iterations=1)

plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dilation, cmap='gray'), plt.title("Dilation")
plt.xticks([]), plt.yticks([])
plt.show()
```

#### Output

![Task 11 Output](task11_output.png)

#### Observation

Dilation with a 3×3 kernel expands bright regions by replacing each pixel with the maximum value in its neighborhood. This fills in small dark gaps and thickens bright structures. The output appears slightly brighter overall compared to the original, as bright areas grow outward.

---

### Task 12: Opening of an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/Lenna.jpg', 0)
kernel = np.ones((3,3), np.uint8)

Opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

plt.figure(figsize=(8,4))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(Opening, cmap='gray')
plt.title("Opening")
plt.xticks([])
plt.yticks([])

plt.show()
```

#### Output

![Task 12 Output](task12_output.png)

#### Observation

Morphological Opening (Erosion followed by Dilation) removes small bright noise spots while approximately preserving the size and shape of larger bright regions. The resulting image is slightly smoother with fine bright protrusions eliminated, which is useful for separating connected objects.

---

### Task 13: Closing of an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/Lenna.jpg', 0)
kernel = np.ones((3,3), np.uint8)

Closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(8,4))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(Closing, cmap='gray')
plt.title("Closing")
plt.xticks([])
plt.yticks([])

plt.show()
```

#### Output

![Task 13 Output](task13_output.png)

#### Observation

Morphological Closing (Dilation followed by Erosion) fills small dark holes within bright regions while approximately maintaining the overall shape and size of objects. The output appears slightly more filled-in compared to the original, effectively bridging small gaps in bright structures.

---

### Task 14: Morphological Gradient of an Image

#### Code

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/content/Lenna.jpg', 0)
kernel = np.ones((3,3), np.uint8)

Gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.figure(figsize=(8,4))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(Gradient, cmap='gray')
plt.title("Gradient")
plt.xticks([])
plt.yticks([])

plt.show()
```

#### Output

![Task 14 Output](task14_output.png)

#### Observation

The Morphological Gradient (Dilation − Erosion) highlights the boundaries between bright and dark regions of the image. The output is a dark background with bright edges marking object contours. This is particularly effective for detecting the outline of structures without requiring explicit edge detection filters like Sobel or Laplacian.

---

## Conclusion

In this lab, various image processing operations related to noise modeling, denoising, and morphological analysis were successfully performed using OpenCV, NumPy, and Matplotlib in Python. Six different types of noise — Gaussian, Rayleigh, Erlang, Exponential, Uniform, and Salt & Pepper — were added to images, each producing characteristically different degradation. Denoising techniques including Gaussian Blur, Median Blur, and Non-Local Means were applied, with Non-Local Means yielding the best reconstruction quality for Gaussian noise and Median Filter proving most effective against Salt & Pepper noise. Morphological operations — Erosion, Dilation, Opening, Closing, and Gradient — were demonstrated, illustrating how structuring elements can be used to remove noise, fill gaps, and detect object boundaries based purely on image shape and structure.
