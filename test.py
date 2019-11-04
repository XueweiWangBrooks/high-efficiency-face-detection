import os
import keras
import fire
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import CustomObjectScope
from keras_applications.mobilenet import relu6
from keras.callbacks import TensorBoard
from keras.preprocessing import image
import cv2


def main(trained_model, test_img_path, pred_model='single'):
    
    print(pred_model)
    label = ['angry', 'disgust', 'happy', 'neutral', 'sad', 'surprised']
    label.sort()


    with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        model = keras.models.load_model(trained_model);
    
    label_num_img = [0,] * len(label)
    label_num_acc = [0,] * len(label)
    label_total_confid = [0,] * len(label)
    overall_dist = []

    for i in range(0, len(label)):

        # distribution of tags predicted
        dist = [0, ] * len(label)
        dir = os.path.join(test_img_path, label[i])
        print("testing on '{}' ...".format(label[i]))
        
        for file in os.listdir(dir):
            img = image.img_to_array(image.load_img(os.path.join(dir, file)))
            
            if pred_model == 'github':
                img = cv2.resize(src=img, dsize=(48, 48))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=2)  
            elif pred_model == 'combined' or 'single':
                img = cv2.resize(src=img, dsize=(128, 128))
                
            img = np.expand_dims(img, axis=0)  
            
            if pred_model == 'combined':      
                # only care about emotion
                pred = model.predict(img)[1]
            elif pred_model == 'github' or 'single':
                pred = model.predict(img)
                # print(pred)
            
            pred_idx = np.argmax(pred, axis=1).item()
            if pred_model == 'github':
                # github model has 7 category. map to 6. index = 2 means fearful, which is not in labels.
                label_num_img[i] += 1
                if pred_idx != 2:    
                    if pred_idx > 2:
                        dist[pred_idx - 1] += 1
                        pred_idx -= 1
                    else:
                        dist[pred_idx] += 1
                    if pred_idx == i:
                        label_num_acc[i] += 1
                
                if i >= 2:
                    label_total_confid[i] += pred[0][i + 1]
                else:
                    label_total_confid[i] += pred[0][i]


            if pred_model == 'combined' or 'single':
                label_num_img[i] += 1
                dist[pred_idx] += 1
                if pred_idx == i:
                    label_num_acc[i] += 1
                label_total_confid[i] += pred[0][i]

            if label_num_img[i] % 100 == 0:
                print("\timage tested = {}, accuracy = {}, average confidence = {}, distribution = {}".format(label_num_img[i], float(label_num_acc[i])/label_num_img[i], float(label_total_confid[i])/label_num_img[i], dist))
        
        overall_dist.append(dist)

    print("{},{},{},{},{}".format("label", "num image", "accuracy", "avg confidence", "distribution"))
    num_total = 0
    total_acc = 0
    total_conf = 0

    for i in range(0, len(label)):
        if(label_num_img[i] > 0):
            print("{},{},{},{},{}".format(label[i], label_num_img[i], float(label_num_acc[i])/label_num_img[i], float(label_total_confid[i])/label_num_img[i], overall_dist[i]))
            num_total += label_num_img[i]
            total_acc += label_num_acc[i]
            total_conf += label_total_confid[i]
    
    print("{},{},{},{},{}".format("total", num_total, float(total_acc)/num_total, float(total_conf)/num_total, "N/A"))
        # load weights into new model
        # model.load_weights(trained_model)
        # model.compile('sgd','mse')
        # datagen = ImageDataGenerator()
        # test_gen = datagen.flow_from_directory(directory=data_path +
        #                                     "/expression/test",
        #                                     target_size=(128, 128),
        #                                     class_mode="categorical",
        #                                     batch_size=128)

        # tensorboard = TensorBoard(log_dir=log_folder,
        #                       histogram_freq=0,
        #                       write_graph=True,
        #                       write_images=True)\
        # only care about the emotion predictions
        # pred = model.predict_generator(test_gen, verbose=1)[1]
        # labels = (test_gen.class_indices)
        # labels = dict((v,k) for k,v in labels.items())
        # print(pred)
        # print(pred.shape)
        # print (labels)

# predicted_class_indices=np.argmax(pred,axis=1)
# labels = (validation_generator.class_indices)
# labels2 = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]

if __name__ == "__main__":
    fire.Fire(main)