# This file is only used to imlement clache on the input file.
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dicom
import os
from PIL import Image
import nibabel as nib
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import modify_image, plot, hu_units
#%%
file = "D:\\Szakdoga\\SortedPictures\\3\\1-15.dcm"

# Desktop/OneDrive_1_17-02-2024/week_4/het4_morfologia_orai.ipynb

dd = dicom.dcmread(file)

image = modify_image(file)

# image = (dd.pixel_array)/16

print(image[100,100])
print(image.shape)

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('on')


plt.subplot(1, 2, 2)
plt.imshow(image[150:250,75:175], cmap='gray')
plt.axis('on')
#%%
path = 'D:\\Szakdoga\\LCI\\100046.nii'

img = nib.load(path).get_fdata()
# img.shape
img2 = img[:,:,52] < 0
plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.axis('on')
#%%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('on')
plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray')
plt.axis('on')

"""
noise cancellation before clahe seems to produce better visibility
h = 4 works best from 5 it is too blurry
"""

# plt.subplot(1, 2, 2)
# plt.title("Noise off and CLAHE")
# plt.imshow(claheImage_denoised, cmap='gray')
# plt.axis('off')


# 
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
np.dtype(modify_image(file_path)[0,0])
#%%
directory = "D:\\Szakdoga\\SortedPictures\\3\\"
images = []
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        masked = modify_image(file_path) & (hu_units(file_path) * 255)
        images.append(masked)

plot(images[1], "test2")
#%% Making images all uniform uint8 and RGB

processed_images = []
for img in images:
    # # Convert float images to 0-255 uint8
    # if img.dtype in [np.float32, np.float64]:
    #     img = (img * 255).clip(0, 255).astype(np.uint8)
    # elif img.dtype != np.uint8:
    #     img = img.astype(np.uint8)

    # Convert grayscale to RGB for compatibility
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=2)

    processed_images.append(img)
#%% Save to GIF
# Convert NumPy arrays to PIL images
pil_images = [Image.fromarray(img) for img in images]

# Save with duration per frame (ms), loop=0 means infinite
pil_images[0].save(
    'output_pillow.gif',
    save_all=True,
    append_images=pil_images[1:],
    duration=150,  # 500ms = 0.5s
    loop=0
)

