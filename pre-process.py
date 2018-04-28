# -*- coding: utf-8 -*-

import tarfile


if __name__ == '__main__':
    tar = tarfile.open('mart/cars_train.tgz', "r:gz")
    tar.extractall()
    tar.close()
