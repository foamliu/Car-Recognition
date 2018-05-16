# import the necessary packages
import cv2 as cv
import numpy as np
import scipy.io
from utils import load_model, draw_str
import argparse
import keras.backend as K
import os
import random


if __name__ == '__main__':
    model = load_model()
    model.load_weights('models/model.96-0.89.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    test_path = 'cars_test/'
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]
    samples = random.sample(test_images, 16)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join('data/test', image_name)
        print('Start processing image: {}'.format(filename))

        orig_img = cv.imread(os.path.join(test_path, image_name))
        bgr_img = cv.imread(filename)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob = np.max(preds)
        class_id = np.argmax(preds)
        text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob))
        draw_str(orig_img, (20, 20), text)
        cv.imwrite('images/{}_out.png'.format(i), orig_img)


    K.clear_session()

