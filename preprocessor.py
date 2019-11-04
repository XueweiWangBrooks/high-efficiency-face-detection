import fire
import cv2
import os
import numpy as np
import random
import math
   # center
        
def crop(image, corner, num_pixel):
    h = image.shape[0]
    w = image.shape[1]
    switcher = {
        0: image,
        1: image[0:h - num_pixel, 0:w - num_pixel],
        2: image[0:h - num_pixel, num_pixel: w],
        3: image[num_pixel: h, 0:w - num_pixel],
        4: image[num_pixel: h, num_pixel: w]
    }
    return switcher.get(corner)
    
def horizontal_flip(image):
    return cv2.flip(image, flipCode=1)

def add_noise(image, sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    return image + gauss

def adjust_contrast(image, alpha):
    return cv2.addWeighted(image, alpha, image, 0, 0)

def blur(image, kernel):
    return cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)

def save_image(image, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path + '/' + filename, image)

def random_preprocess(image):
    corner = random.randint(0, 4)
    num_pixel = random.randint(0, 10)
    image = crop(image, corner, num_pixel)
    flip_float = random.random()
    if(flip_float > 0.5):
        image = horizontal_flip(image)
    sigma = random.expovariate(10)
    image = add_noise(image, sigma)
    alpha = random.uniform(0.5, 1.2)
    image = adjust_contrast(image, alpha)
    kernel_size = random.expovariate(2)
    image = blur(image, (kernel_size, kernel_size))
    return image

def main(data_path, target_path):
    for dirpath, dirname, filenames in os.walk(data_path):
        for filename in filenames:
            if ".jpg" in filename:
                print(filename)
                image = cv2.imread(dirpath + '/' + filename)
                image = random_preprocess(image)
                save_image(image, target_path + '/' + dirpath, filename)

if __name__ == "__main__":
    fire.Fire(main)