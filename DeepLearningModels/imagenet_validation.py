'''Run after the model is trained
   output validation result as html
'''

import os
import glob
import numpy as np
from keras.preprocessing import image
import inception_v3 as inception


IMSIZE = (299, 299)

def ValidateAllImgs(model):
    base_path = './Data/sport3/validation/'
    for dirnames in os.listdir(base_path):
        dirpath = os.path.join(base_path, dirnames)
        img_paths = glob.glob(os.path.join(dirpath, "*.jpg"))
        for img_path in img_paths:
            img = image.load_img(img_path, target_size=IMSIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = inception.preprocess_input(x)
            preds = model.predict(x)
            print('Predicted:', preds)

def ValidateImg(model, img_path):
    img = image.load_img(img_path, target_size=IMSIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = inception.preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', preds)

# img = image.load_img(img_path, target_size=IMSIZE)
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# x = inception.preprocess_input(x)

# preds = model.predict(x)
# print('Predicted:', preds)
