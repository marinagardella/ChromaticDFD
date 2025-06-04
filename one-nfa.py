import numpy as np
from scipy.stats import binom, norm
from skimage.filters import threshold_multiotsu
from scipy import ndimage
import cv2
import os
from imutils.object_detection import non_max_suppression
import argparse
from pathlib import Path
import skimage.io as io

PATH = '/Users/julietaumpierrez/Desktop/iccv-doc-workshop/ChromaticDFD/ChromaticFD-outputs/original'

for folder in os.listdir(PATH):
    nfa1 = io.imread(os.path.join(PATH, folder, 'nfa_0.png'))/255
    nfa2 = io.imread(os.path.join(PATH, folder, 'nfa_1.png'))/255
    nfa3 = io.imread(os.path.join(PATH, folder, 'nfa_2.png'))/255
    output = nfa1 * nfa2 * nfa3
    cv2.imwrite(os.path.join(PATH, folder, 'nfa.png'), (output * 255))  
