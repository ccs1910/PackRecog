
import sys
import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model

from test import test

import os

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--image", help="path to image")
    args = a.parse_args()

    test = test()

    print("\n1. Test the model")
    model = test.model

    print("\n2. Load the To be Predicted Image")
    # img_path = '/Users/csantoso/AppData/Local/My Private Documents/_BitBucket/DL_python/pack_rec/PackRecognition/rokok_small/test/IQOS GREEN - Marlboro Menthol/IMG_9354.JPG'
    img_path = '/Users/csantoso/AppData/Local/My Private Documents/_BitBucket/DL_python/pr/PackRecog/rokok_small/others/trial3.JPG'
# C:\Users\csantoso\AppData\Local\My Private Documents\_BitBucket\DL_python\pr\PackRecog\rokok_small\others\m1.jpg
    img = image.load_img(img_path, target_size=(150,150))
    
    print("\n3. Convert image to Array")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # print("x data : ",x)
    x = x.astype('float32')
    x /= 255
    x = np.where(x > 0.5, 1, 0)
    # print("x data - Zero Centered : ",x)

    print("\n4. Create List of Labels from the test directory")
    labels_dict = test.labels_dict 
    labels_list = []

    # reversed the class indices
    reversed_labels_dict = dict(zip(labels_dict.values(), labels_dict.keys()))
   
    # create the list of labels
    for k in sorted(reversed_labels_dict.keys()) :
        labels_list.append(reversed_labels_dict[k])


    print("\n5. Predict the Image")
    preds = model.predict(x)


    # PLOT Prediction

    print('\n6. Show Image')

    plt.imshow(img)

    plt.axis('off')

    plt.figure()


    print("\n7. Show Probability")

    bar_preds = [pr for pr in preds]
    pred_dict = dict(zip(labels_list,bar_preds[0].tolist() ))

    plt.barh(range(len(pred_dict)), list(pred_dict.values()), align='center')
    plt.yticks(range(len(pred_dict)), list(pred_dict.keys()))

    plt.xlabel('Probability')
    plt.xlim(0, 1.01)

    # for i, v in enumerate(list(pred_dict.values())):
    #     plt.text(v + 3, i + .25, str(v), color='red', fontweight='bold')

    plt.tight_layout()
    
    plt.show()
