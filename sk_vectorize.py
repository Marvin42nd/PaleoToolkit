# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:59:16 2023

@author: kelle
"""

import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import cv2

      
vector_pic = np.array(imread('./Pictures/Paul 3.jpg'))
plt.imshow(vector_pic)