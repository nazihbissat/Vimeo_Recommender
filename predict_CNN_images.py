from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import os
from os.path import isfile, join
import numpy as np
import pickle


def cnn_imagenet_predict(folder_path):
    """ Return list of vectorized images"""

    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    image_files = [f for f in os.listdir(folder_path) if (isfile(join(folder_path, f)) and f != ".DS_Store")]
    images = []
    for pic in image_files:
        img = image.load_img((folder_path + "/" + pic), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)[0]
        images += [(pic.split("_")[0], np.char.mod('%f', features))]
    return images
