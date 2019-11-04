import cv2
import fire
import os
import re
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 


def traverse_folder(folder_path, out_folder, video_format, overwrite, skip_on_label):
    retval = []
    for dir_name, subdir_list, file_list in os.walk(folder_path):
        for file in file_list:
            if str('.' + video_format) in file:
                tags = re.compile(r"\-|\.").split(file)
            
            target = dir_name + "/" + file;
            
            if not overwrite:
                if os.path.exists(out_folder + target.replace(folder_path, "")):
                    print("folder exists: {}. To overwrite, set --overwrite=true".format(target))
                    continue
            
            if skip_on_label:
                # skip fearful(06) and calm(02)
                if tags[0] == '01':
                    if not (tags[2] == '02' and tags[2] == '06'):
                        retval.append(target)
            else:
                retval.append(target)

    return retval

# skip first and last n second to capture the actual emotion rather than the actor preparing.
def capture_frames(video, interval, n):
    imgs = []
    vidcap = cv2.VideoCapture(video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    num_frame = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    skip_count = fps * n

    success, image = vidcap.read()
    skip_count -= 1
    while(success and skip_count > 0):
        success, new_img = vidcap.read()
        skip_count -= 1
    
    dur_count = num_frame - fps * 2 * n
    intv_count = 0
    while(success and dur_count > 0):
        success, new_img = vidcap.read()
        intv_count += 1
        dur_count -= 1
        if (intv_count == int(interval)):
            intv_count = 0
            imgs.append(image)
            
            # print(len(image))
            image = new_img
    return imgs

def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces
    face_rect = face_cascade.detectMultiScale(image = gray, scaleFactor=1.05, minNeighbors = 5, minSize=(50,50))
    faces = []

    # no face found
    if len(face_rect) == 0:
        return faces

    # only retrive the largest
    face = -1
    largest_size = 0 
    (x,y,w,h) = (0,0,0,0)
    for (x_t,y_t,w_t,h_t) in face_rect:
        size = w_t * h_t
        if(size > largest_size):
            largest_size = size
            (x,y,w,h) = (x_t,y_t,w_t,h_t)

    # print("face found: {}".format((x,y,w,h)))  

    faces.append(image[y-20:y+h+20, x-20:x+w+20])
    return faces

def resize_image(image, dim):
    if image is None:
        return None
    return cv2.resize(src=image, dsize=(int(dim[0]), int(dim[1])))

def save_image(image, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    cv2.imwrite(dir + '/' + name + '.jpg', image)

def process_single_video(video, in_folder, out_folder, cropped_dim, frame_interval, video_format, skip_seconds, skip_on_label):
    dim = cropped_dim.split(',')
    print('processing video.. filename = %s' % video)
    img_list = capture_frames(video, frame_interval, skip_seconds)
    target_dir = out_folder + video.replace(in_folder, "")
    
    count = 0
    for img in img_list:
        # save_image(img, target_dir, str(count))
        faces = crop_face(img)
        for face in faces:
            if face is None:
                print("abnormal: face located very near image boundaries.")
                continue
            face = resize_image(face, dim)
            save_image(face, target_dir, str(count))
            count += 1

    print('num images saved: {}. location: {}'.format(count, target_dir))
    return count

#cropped_dim: resized len, resized hight, crop x start, crop x end, crop y start, crop y end.
def main(in_folder='./data_raw', out_folder='./data_normalized', cropped_dim="128,128", frame_interval="5", video_format="mp4", overwrite=False, skip_seconds = 1, skip_on_label=False):
    # print('cv2 version: ' + cv2.__version__)
    video_list=traverse_folder(in_folder, out_folder, video_format, overwrite, skip_on_label)
    # print(video_list)
    
    # process_single_video(video_list[0], in_folder, max_num_file, out_folder, cropped_dim, frame_interval, video_format)

    for video in video_list:
        process_single_video(video, in_folder, out_folder, cropped_dim, frame_interval, video_format, skip_seconds, skip_on_label)


if __name__ == "__main__":
    fire.Fire(main)

