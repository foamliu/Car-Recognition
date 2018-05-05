# import the necessary packages
import cv2 as cv
import numpy as np
import scipy.io
from utils import load_model
import argparse


if __name__ == '__main__':
    model = load_model()

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    args = vars(ap.parse_args())

    filename = args["image"]
    if filename is None:
        filename = 'images/samples/04732.jpg'

    bgr_img = cv.imread(filename)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    class_id = np.argmax(preds)
    print('class_name: ' + class_names[class_id])
    print('prob: ' + str(prob))
