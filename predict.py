import sys
import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model

import os

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--image", help="path to image")
    args = a.parse_args()

    model = load_model('cigarettes1.h5')
    model.summary()

    # img_path = '/Users/csantoso/AppData/Local/My Private Documents/_BitBucket/DL_python/pack_rec/PackRecognition/rokok_small/test/IQOS GREEN - Marlboro Menthol/IMG_9354.JPG'
    img_path = '/Users/csantoso/AppData/Local/My Private Documents/_BitBucket/DL_python/pack_rec/PackRecognition/rokok_small/others/m1.JPG'

    img = image.load_img(img_path, target_size=(150,150))
    

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print("x data : ",x)
    x = x.astype('float32')
    x /= 255
    x = np.where(x > 0.5, 1, 0)
    print("x data - Zero Centered : ",x)

    preds = model.predict(x)

    print('Predicted:', preds)

    # PLOT Prediction

    print('Show Image')

    plt.imshow(img)

    plt.axis('off')

    plt.figure()


    print("Show Probability")

    #get folder list and put the score

    base_dir = '/Users/csantoso/AppData/Local/My Private Documents/_BitBucket/DL_python/pack_rec/PackRecognition/rokok_small'
    test_dir = os.path.join(base_dir, 'test')

    bar_preds = [pr for pr in preds]

    pred_dict = dict(zip(os.listdir(test_dir),bar_preds[0].tolist() ))

    print("pred_dict",pred_dict, "SHOW IT")

    plt.barh(range(len(pred_dict)), list(pred_dict.values()), align='center')
    plt.yticks(range(len(pred_dict)), list(pred_dict.keys()))

    plt.xlabel('Probability')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    
    plt.show()

