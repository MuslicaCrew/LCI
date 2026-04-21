# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:45:53 2025

@author: 360br
"""

# This file is only used to imlement clache on the input file.
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import os
from PIL import Image
import nibabel as nib
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import modify_image, plot


file = "D:\\Szakdoga\\SortedPictures\\3\\1-15.dcm"

# Desktop/OneDrive_1_17-02-2024/week_4/het4_morfologia_orai.ipynb

dd = dicom.dcmread(file)

image = dd.pixel_array

slope = dd.RescaleSlope
intercept = dd.RescaleIntercept

hu_image = image * slope + intercept
# Create lung mask based on HU thresholds
lung_mask = ((hu_image > -950) & (hu_image < -400)).astype(np.uint8)

# Plot original image and lung mask
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image, cmap='gray', vmin=-2000, vmax=2000)
axs[0].set_title("CT image")
axs[0].axis("off")

axs[1].imshow(lung_mask, cmap='gray')
axs[1].set_title("Extracted Lung Mask (-950 < HU < -400)")
axs[1].axis("off")

plt.tight_layout()
plt.show()
#%%
print("Rescale Slope:", dd.get("RescaleSlope", "N/A"))
print("Rescale Intercept:", dd.get("RescaleIntercept", "N/A"))
print("Modality:", dd.get("Modality", "N/A"))
print("Pixel Range:", dd.pixel_array.min(), "to", dd.pixel_array.max())
#%%

dcm = dicom.dcmread(file)
image = dcm.pixel_array.astype(np.int16)

slope = dcm.RescaleSlope
intercept = dcm.RescaleIntercept

print(f"Original pixel range: {image.min()} to {image.max()}")
hu_image = image * slope + intercept
print(f"Converted to HU range: {hu_image.min()} to {hu_image.max()}")