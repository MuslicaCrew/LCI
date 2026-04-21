# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:07:55 2025

@author: 360br
"""
import numpy as np
import pydicom as dicom
import cv2
import matplotlib.pyplot as plt
#%%


def modify_image(file):
   dicom_data = dicom.dcmread(file)
   image_data = dicom_data.pixel_array
   image_data -= np.min(image_data)
   image_new = image_data / np.max(image_data)
   image_new *= 255
   image_uint8 = image_new.astype(np.uint8) 
   claheObj = cv2.createCLAHE(clipLimit=5)
   claheImage = claheObj.apply(image_uint8)
   denoised = cv2.fastNlMeansDenoising(claheImage, None, h=4, templateWindowSize=7, searchWindowSize=21)
   claheImage_denoised = claheObj.apply(denoised)
   return claheImage_denoised

def plot(image, title):
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    return;
    
def hu_units(file):
    dicom_data = dicom.dcmread(file)
    image = dicom_data.pixel_array
    slope = dicom_data.RescaleSlope
    intercept = dicom_data.RescaleIntercept
    hu_image = image * slope + intercept
    lung_mask = ((hu_image > -950) & (hu_image < -400)).astype(np.uint8)
    return lung_mask