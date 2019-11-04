import os
import fire
import re
import random
from shutil import copy2

# fer_emo_list = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
class Catagory_maker(object):

    def __init__(self):
        self.folders = ('train', 'test', 'validation')
        self.labels = ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised')
        self.total_num_img = 0
        self.num_img = [0,] * 3

    # return the folder index randomly assigned for this img.
    def rand_assign_folder(self):
        retval = 2
        rand = random.uniform(0,1)
        if rand < float(self.num_img[0])/self.total_num_img:
            self.num_img[0] -= 1
            retval = 0
        elif rand < float(self.num_img[0])/self.total_num_img + float(self.num_img[1])/self.total_num_img:
            self.num_img[1] -= 1
            retval = 1
        else:
            self.num_img[2] -= 1

        # decrease afterwards to prevent divide by zero err.
        self.total_num_img -= 1
        return retval

    def make_dirs(self, root):
        for folder in self.folders:
            for label in self.labels:
                    # skip fearful and calm
                    if label == 'fearful' or label == 'calm':
                        continue
                    os.makedirs(root + "/expression/" + folder + "/" + label)
    
    def move_file(self, file_path, root, purpose, folder, label, file_name):
        # print('{}/{}/{}/{}/{}'.format(root, purpose, folder, label, file_name))
        copy2(file_path, '{}/{}/{}/{}/{}'.format(root, purpose, folder, label, file_name))

    def count_image(self, root, image_format):
        num = 0
        for dir_name, subdir_list, file_list in os.walk(root):
            for file in file_list:
                if str('.' + image_format) in file:
                    num += 1
        print('total images found: ' + str(num))
        self.total_num_img = num
        self.num_img[0] = int(self.total_num_img * 0.7)
        self.num_img[1] = int(self.total_num_img * 0.15)
        self.num_img[2] = self.total_num_img - self.num_img[0] - self.num_img[1]
        print('images to train: {}, to test: {}, to validate: {}'.format(self.num_img[0], self.num_img[1], self.num_img[2]))


def main(target_path='./data', image_format='jpg', data_path='./data_normalized', dataset = 'RAVD'):
    cm = Catagory_maker()
    cm.count_image(data_path, image_format)
    
    label_count = [0,] * 8;
    cm.make_dirs(target_path)

    # walk through all images
    for dir_name, subdir_list, file_list in os.walk(data_path):
        for file in file_list:
            if str('.' + image_format) in file:    

                # interpret the tag
                if dataset == 'RAVD':
                    dir_tree = re.compile(r"/|\\").split(dir_name)
                    descript = dir_tree[-1]
                    tags = re.compile(r"\-|\.").split(descript)
                    tag_idx = int(tags[2]) - 1
                
                if dataset == 'FER':
                    dir_tree = re.compile(r"/|\\").split(dir_name)
                    fer_tag = int(dir_tree[1])

                    # fer_emo_list = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
                    # cm.labels = ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised')
                    # mapping from fer_tag to tag
                    tag_map = (4, 6, 5, 2, 3, 7, 0)
                    tag_idx = tag_map[fer_tag]
                
                tag = cm.labels[tag_idx]
                
                # skip fearful(5) and calm(1)
                if(tag_idx == 5 or tag_idx == 1):
                    continue
                
                folder_idx = cm.rand_assign_folder()
                folder = cm.folders[folder_idx]
                
                # file_path, root, purpose, folder, label, file_name
                cm.move_file(
                    dir_name + '/' + file,
                    target_path, 
                    'expression', 
                    folder,
                    tag,
                    str(cm.num_img[folder_idx]) + '.' + image_format
                    )

                label_count[tag_idx] += 1

                if cm.total_num_img % 100 == 0:
                    print("labels found ({}): {}".format(cm.labels, label_count))
                    print("remaining: img = {}, train = {}, test = {}, val = {}".format(cm.total_num_img, cm.num_img[0], cm.num_img[1], cm.num_img[2]))

if __name__ == "__main__":
    fire.Fire(main)