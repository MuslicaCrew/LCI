# This file is only used to imlement clache on the input file.
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import cv2
from skimage.restoration import denoise_tv_chambolle
import os


file = "D:\\Szakdoga\\SortedPictures\\3\\1-15.dcm"
dicom_data = dicom.dcmread(file)
image_data = dicom_data.pixel_array/1
image_data -= np.min(image_data)
image_new = image_data / np.max(image_data)
image_new *= 255
image_uint8 = image_new.astype(np.uint8)

claheObj = cv2.createCLAHE(clipLimit=5)
claheImage = claheObj.apply(image_uint8)
denoised = cv2.fastNlMeansDenoising(claheImage, None, h=4, templateWindowSize=7, searchWindowSize=21)
img_float = claheImage / 255.0
denoised_tv = denoise_tv_chambolle(img_float, weight=0.01)
denoised_uint8 = (denoised_tv * 255).astype(np.uint8)
claheImage_denoised = claheObj.apply(denoised)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.title("Original")
# plt.imshow(image_uint8, cmap='gray')
# plt.axis('off')

plt.subplot(1, 2, 1)
plt.title("After CLAHE")
plt.imshow(claheImage, cmap='gray')
plt.axis('off')

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

plt.subplot(1, 2, 2)
plt.title("Noise off and CLAHE")
plt.imshow(claheImage_denoised, cmap='gray')
plt.axis('off')


plt.show()