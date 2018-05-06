# import the necessary packages
import cv2 as cv
import numpy as np
import scipy.io
from utils import load_model
import argparse
import keras.backend as K


if __name__ == '__main__':
    model = load_model()
    model.load_weights('models/model.96-0.89.hdf5')

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    args = vars(ap.parse_args())

    filename = args["image"]
    if filename is None:
        filename = 'images/samples/07647.jpg'

    bgr_img = cv.imread(filename)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    class_id = np.argmax(preds)
    print('class_name: ' + str(class_names[class_id][0][0]))
    print('prob: ' + str(prob))

    K.clear_session()

