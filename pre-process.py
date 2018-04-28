# -*- coding: utf-8 -*-

import tarfile
import scipy.io
import numpy as np
import os
import shutil
import random


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_data(usage, fnames, labels):
    src_folder = 'cars_{}'.format(usage)
    num_samples = len(fnames)

    if (usage == 'train'):
        num_train = int(round(num_samples * 0.8))
        num_valid = num_samples - num_train
        train_indexes = random.sample(num_samples, num_valid)

    for i in range(num_samples):
        fname = fnames[i]
        label = labels[i][0]
        print("{} -> {}".format(fname, label))
        src_path = os.path.join(src_folder, fname)
        if (usage == 'train'):
            if i in train_indexes:
                dst_folder = 'data/train'
            else:
                dst_folder = 'data/valid'
        else:
            dst_folder = 'data/{}'.format(usage)

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)
        shutil.move(src_path, dst_path)


def process_data(usage):
    print("Processing {} data...".format(usage))
    cars_annos = scipy.io.loadmat('devkit/cars_{}_annos'.format(usage))
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    labels = []
    try:
        for annotation in annotations:
            #bbox_x1 = annotation[0][0][0][0]
            #bbox_y1 = annotation[0][1][0][0]
            #bbox_x2 = annotation[0][2][0][0]
            #bbox_y2 = annotation[0][3][0][0]
            class_id = annotation[0][4][0][0]
            class_name = class_names[class_id][0]
            fname = annotation[0][5][0]
            #print("bbox_x1={}, bbox_y1={}, bbox_x2={}, bbox_y2={}, class_id={}, class_name={}, fname={}".format(bbox_x1, bbox_y1, bbox_x2, bbox_y2, class_id, class_name, fname))
            fnames.append(fname)
            labels.append(class_name)
    except IndexError as err:
        print(annotation)

    save_data(usage, fnames, labels)


if __name__ == '__main__':

    print('Extracting cars_train.tgz...')
    with tarfile.open('cars_train.tgz', "r:gz") as tar:
        tar.extractall()
    print('Extracting cars_test.tgz...')
    with tarfile.open('cars_test.tgz', "r:gz") as tar:
        tar.extractall()
    print('Extracting car_devkit.tgz...')
    with tarfile.open('car_devkit.tgz', "r:gz") as tar:
        tar.extractall()

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    print(class_names[8][0])

    ensure_folder('data/train')
    ensure_folder('data/valid')
    ensure_folder('data/test')

    process_data('train')
    process_data('test')

    # clean up
    shutil.rmtree('cars_train')
    shutil.rmtree('cars_test')
    shutil.rmtree('devkit')




