import numpy as np
from scipy.stats import binom, norm
from skimage.filters import threshold_multiotsu
from scipy import ndimage
import cv2
import argparse

def load_image(img_path):
    """
    Loads color image from a given path
    """
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    return img

def get_chromatic_image(img):
    """
    Computes the chromatic image of the input image
    """
    grey_mean = np.mean(img, axis=-1)
    chroma = img.copy()
    chroma[:,:,0] = img[:,:,0]- grey_mean 
    chroma[:,:,1] = img[:,:,1] - grey_mean 
    chroma[:,:,2] = img[:,:,2] - grey_mean 
    return chroma

def binarize_image(img, nb_classes=3):
    """
    Binarizes a grayscale image using the smallest threshold derived from  
    the multi-otsu method with nb_classes classes
    """
    thresh = threshold_multiotsu(img, classes = nb_classes)[0]
    return img < thresh

def extract_components(img, blob_thresh=10):
    """
    Extracts labeled connected components from the binarized image, removing  
    components with area smaller than blob_thresh
    """
    binary= binarize_image(img)
    # Label connected components
    labeled_objects, _ = ndimage.label(binary)
    
    # Remove components whose area is smaller than blob_thresh 
    counts = np.bincount(labeled_objects.ravel())
    valid_labels = np.where(counts > blob_thresh)[0]
    valid_labels = valid_labels[1:]
    # Create a mask with only the valid labels
    valid_objects = np.isin(labeled_objects, valid_labels) * labeled_objects

    return valid_objects, valid_labels.tolist()

def compute_std_per_label(img, labeled_objects, labels):
    """
    Computes standard deviation for each labeled region 
    """
    stds = ndimage.labeled_comprehension(img, labeled_objects, labels, np.std, float, 0)

    # Create an image with standard deviations assigned to each region
    stds_img = np.zeros_like(img, dtype=np.float32)
    for i, std in zip(labels, stds):
        stds_img[labeled_objects == i] = std
    
    stds_img -= stds_img.min()
    stds_img /= stds_img.max()

    return stds, stds_img

def compute_outliers_mask(label_objects, labels, stds, alpha=0.1):
    """
    Computes a statistical mask using standard deviation filtering.
    """
    mean_stds = np.mean(stds)
    sigma_stds = np.std(stds)
    z = norm.isf(alpha)
    thresh = mean_stds - z * sigma_stds

    # Create mask where standard deviation is below threshold
    mask = np.isin(label_objects, [i for i, std in zip(labels, stds) if std < thresh]).astype(np.uint8) * 255

    return mask

def compute_NFA(mask, chars, text_mask, threshold, alpha):
    label_words, nb_words = ndimage.label(text_mask)
    NFADets = np.zeros_like(text_mask, dtype=np.uint8)
    for label in range(1, nb_words+1):
        word_mask = (label_words == label)
        _, tot = ndimage.label(word_mask & chars)
        _, dets = ndimage.label(word_mask & mask)
        NFA = nb_words * (1 - binom.cdf(dets-1, tot, alpha))
        if NFA < threshold:
            NFADets[(word_mask & chars)>0] = 1
    return NFADets

def detect_nonparallelized(img_path, mask_path, alpha, threshold):
    img = load_image(img_path)
    chroma_img = get_chromatic_image(img)
    for ch in range(3):
        img_ch = img[:,:,ch]
        chroma_ch = chroma_img[:,:,ch]
        labeled_objects, labels = extract_components(img_ch)
        chars = (labeled_objects > 0)
        stds, stds_img = compute_std_per_label(chroma_ch, labeled_objects, labels)
        outliers = compute_outliers_mask(labeled_objects, labels, stds, alpha)
        text_mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        nfa = compute_NFA(outliers, chars, text_mask, threshold, alpha)
        cv2.imwrite(f'characters_{ch}.png', chars * 255)
        cv2.imwrite(f'outliers_{ch}.png', outliers)
        cv2.imwrite(f'nfa_{ch}.png', nfa * 255)
        cv2.imwrite(f'stds_{ch}.png', stds_img * 255)

from concurrent.futures import ThreadPoolExecutor

def process_channel(ch, img, chroma_img, mask_path, alpha, threshold):
    """Processes a single channel in parallel."""
    img_ch = img[:, :, ch]
    chroma_ch = chroma_img[:, :, ch]
    
    labeled_objects, labels = extract_components(img_ch)
    chars = (labeled_objects > 0)
    
    stds, stds_img = compute_std_per_label(chroma_ch, labeled_objects, labels)
    outliers = compute_outliers_mask(labeled_objects, labels, stds, alpha)
    
    text_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    nfa = compute_NFA(outliers, chars, text_mask, threshold, alpha)

    # Save output images
    cv2.imwrite(f'characters_{ch}.png', chars * 255)
    cv2.imwrite(f'outliers_{ch}.png', outliers)
    cv2.imwrite(f'nfa_{ch}.png', nfa * 255)
    cv2.imwrite(f'stds_{ch}.png', stds_img * 255)
    

def detect_parallelized(img_path, mask_path, alpha, threshold):
    """Detects outliers in an image using parallel processing across three channels."""
    img = load_image(img_path)
    chroma_img = get_chromatic_image(img)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_channel, ch, img, chroma_img, mask_path, alpha, threshold) 
                   for ch in range(3)]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("mask")
    parser.add_argument("-a")
    parser.add_argument("-t")
    parser.parse_args()
    args = parser.parse_args()
    img_path = args.image
    mask_path = args.mask
    alpha = float(args.a)
    trheshold = float(args.t)
    detect_parallelized(img_path, mask_path, alpha, trheshold)