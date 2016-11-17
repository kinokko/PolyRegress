'''Run after the model is trained
   output validation result as html
'''

import os
import glob
import numpy as np
from keras.preprocessing import image
import inception_v3 as inception
import html_table_generator as output_generator


IMSIZE = (299, 299)

def ValidateAllImgs(model):
    test_dir = './Data/sport3/validation/'
    for dirnames in os.listdir(test_dir):
        dirpath = os.path.join(test_dir, dirnames)
        img_paths = glob.glob(os.path.join(dirpath, "*.jpg"))
        for img_path in img_paths:
            img = image.load_img(img_path, target_size=IMSIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = inception.preprocess_input(x)
            preds = model.predict(x)
            result = [img_path]
            result.extend(preds[0])
            output_generator.AddData(result)
            # print('Predicted:', preds)
    output_generator.ProcessToHtml()

def ValidateImg(model, img_path):
    img = image.load_img(img_path, target_size=IMSIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = inception.preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', preds)