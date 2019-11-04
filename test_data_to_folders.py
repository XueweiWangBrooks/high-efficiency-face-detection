import os
import fire
import re
import random
from shutil import copy2

class Catagory_maker(object):

    def __init__(self):
        self.labels = ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised')
        self.total_num_img = 0

    def make_dirs(self, root):
        for label in self.labels:
                # skip fearful and calm
                if label == 'fearful' or label == 'calm':
                    continue
                os.makedirs(root + "/" + label)
    
    def move_file(self, file_path, root, label, file_name):
        # print('{}/{}/{}/{}/{}'.format(root, purpose, folder, label, file_name))
        copy2(file_path, '{}/{}/{}'.format(root, label, file_name))

    def count_image(self, root, image_format):
        num = 0
        for dir_name, subdir_list, file_list in os.walk(root):
            for file in file_list:
                if str('.' + image_format) in file:
                    num += 1
        print('total images found: ' + str(num))
        self.total_num_img = num
        print('images to test: {}'.format(self.total_num_img))


def main(target_path='./test', image_format='jpg', data_path='./data_normalized'):
    cm = Catagory_maker()
    cm.count_image(data_path, image_format)
    
    label_count = [0,] * 8;
    cm.make_dirs(target_path)
    counter = 0
    
    # walk through all images
    for dir_name, subdir_list, file_list in os.walk(data_path):
        for file in file_list:
            if str('.' + image_format) in file:    

                # read the folder name, to see if it matchs any of the labels.
                dir_tree = re.compile(r"/|\\").split(dir_name)
                folder_name = dir_tree[-1]
                tag = ""
                tag_idx = -1
                for label in cm.labels:
                    if label in folder_name:
                        tag = label
                        tag_idx = cm.labels.index(label)
                        break
                if tag == "":
                    print("did not find tag in folder: {}, skipped.".format(dir_name))
                    continue
                
                # file_path, root, label, file_name
                cm.move_file(
                    dir_name + '/' + file,
                    target_path, 
                    tag,
                    str(counter) + '.' + image_format
                    )

                counter += 1
                label_count[tag_idx] += 1

                if cm.total_num_img % 100 == 0:
                    print("labels found ({}): {}".format(cm.labels, label_count))
                    print("remaining: img = {}".format(cm.total_num_img))

if __name__ == "__main__":
    fire.Fire(main)