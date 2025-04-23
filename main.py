# This file is only used to imlement clache on the input file.
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pydicom as dicom
import cv2
from skimage.restoration import denoise_tv_chambolle
import os
from PIL import Image
#%%
file = "D:\\Szakdoga\\SortedPictures\\3\\1-15.dcm"

#%%
def modify_image(file):
   dicom_data = dicom.dcmread(file)
   image_data = dicom_data.pixel_array/16
   image_data -= np.min(image_data)
   image_new = image_data / np.max(image_data)
   image_new *= 255
   image_uint8 = image_new.astype(np.uint8) 
   claheObj = cv2.createCLAHE(clipLimit=5)
   claheImage = claheObj.apply(image_uint8)
   denoised = cv2.fastNlMeansDenoising(claheImage, None, h=4, templateWindowSize=7, searchWindowSize=21)
   claheImage_denoised = claheObj.apply(denoised)
   return claheImage_denoised
#%%
# dicom_data = dicom.dcmread(file)
# image_data = dicom_data.pixel_array/16
# image_data -= np.min(image_data)
# image_new = image_data / np.max(image_data)
# image_new *= 255
# image_uint8 = image_new.astype(np.uint8)

# claheObj = cv2.createCLAHE(clipLimit=5)
# claheImage = claheObj.apply(image_uint8)
# denoised = cv2.fastNlMeansDenoising(image_uint8, None, h=4, templateWindowSize=7, searchWindowSize=21)
# img_float = claheImage / 255.0
# denoised_tv = denoise_tv_chambolle(img_float, weight=0.01)
# denoised_uint8 = (denoised_tv * 255).astype(np.uint8)
# claheImage_denoised = claheObj.apply(denoised)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.title("Original")
# plt.imshow(image_uint8, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 1)
# plt.title("After CLAHE")
# plt.imshow(claheImage, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("After NLMD")
# plt.imshow(denoised, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("After tv_chambolle")
# plt.imshow(denoised_uint8, cmap='gray')
# plt.axis('off')

"""
noise cancellation before clahe seems to produce better visibility
h = 4 works best from 5 it is too blurry
"""

# plt.subplot(1, 2, 2)
# plt.title("Noise off and CLAHE")
# plt.imshow(claheImage_denoised, cmap='gray')
# plt.axis('off')


# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("image2")
# plt.imshow(image2, cmap='gray')
# plt.axis('off')



# plt.show()
#%%
directory = "D:\\Szakdoga\\SortedPictures\\3\\"
images = []
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
       images.append(modify_image(file_path))
        

#%%

# Let's say you have a list of NumPy arrays (images)
# images = [claheImage_denoised, denoised_uint8, denoised_tv]  # replace these with your actual image arrays

processed_images = []
for img in images:
    # Convert float images to 0-255 uint8
    if img.dtype in [np.float32, np.float64]:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Convert grayscale to RGB for compatibility
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=2)

    processed_images.append(img)

# Save to GIF
# Convert NumPy arrays to PIL images
pil_images = [Image.fromarray(img) for img in processed_images]

# Save with duration per frame (ms), loop=0 means infinite
pil_images[0].save(
    'output_pillow.gif',
    save_all=True,
    append_images=pil_images[1:],
    duration=150,  # 500ms = 0.5s
    loop=0
)

#%%

#%%