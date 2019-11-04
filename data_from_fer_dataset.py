import csv
import cv2
import numpy as np
import os

emo_list = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
with open("fer2013.csv") as csvfile:
    if not os.path.exists("data_fer_raw"):
        os.mkdir("data_fer_raw")
    for i in range(0,7):
        if not os.path.exists(os.path.join("data_fer_raw", str(i))):
            os.mkdir(os.path.join("data_fer_raw", str(i)))
    readCSV = csv.reader(csvfile, delimiter = ',')
    num_line = 0
    img_count = [0, ] * 7
    for row in readCSV:
        if num_line == 0:
            num_line += 1
            continue
        num_line += 1
        emotion = int(row[0])
        pixels = row[1]
        pixel_list = pixels.split()
        pixel_list = [int(i) for i in pixel_list]
        pixel_list = np.asarray(pixel_list)
        pixel_list = np.reshape(pixel_list, (48, 48))
        pixel_list = pixel_list.astype(np.uint8)
        # pixel_list = np.expand_dims(pixel_list, axis=2)
        image = cv2.resize(pixel_list, (128,128))
        path = os.path.join(".", "data_fer_raw", str(emotion), str(img_count[emotion]) + ".jpg")
        success = cv2.imwrite(path, image)
        if not success:
            print("Error occurs while writing to file.")
        img_count[emotion] += 1
        if (num_line%1000 == 0):
            print("image saved({}): {}".format(emo_list, img_count))
