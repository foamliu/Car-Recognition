# Car Recognition


This repository is to do car recognition by fine-tuning ResNet-152 with Cars Dataset from Stanford.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/random.jpg)

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd Car-Recognition
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

## ImageNet Pretrained Models

Download [ResNet-152](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing) into models folder.

## Usage

### Data Pre-processing
Extract 8,144 training images, and split them by 80:20 rule (6,515 for training, 1,629 for validation):
```bash
$ python pre_process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/train.jpg)

### Analysis
Update "model_weights_path" in "utils.py" with your best model, and use 1,629 validation images for result analysis:
```bash
$ python analyze.py
```

#### Validation acc:
**88.70%**

#### Confusion matrix:

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/confusion_matrix.jpg)

### Test
```bash
$ python test.py
```

Submit predictions of test data set (8,041 testing images) at [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), evaluation result:

#### Test acc:
**88.88%**

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/test.jpg)

### Demo
Download [pre-trained model](https://github.com/foamliu/Car-Recognition/releases/download/v1.0/model.96-0.89.hdf5) into "models" folder then run:

```bash
$ python demo.py --i [image_path]
```
If no argument, a sample image is used:

 ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/samples/07647.jpg)

```bash
$ python demo.py
class_name: Lamborghini Reventon Coupe 2008
prob: 0.9999994
```

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/0_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/1_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/2_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/3_out.png) |
|Hyundai Azera Sedan 2012, prob: 0.99|Hyundai Genesis Sedan 2012, prob: 0.9995|Cadillac Escalade EXT Crew Cab 2007, prob: 1.0|Lamborghini Gallardo LP 570-4 Superleggera 2012, prob: 1.0|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/4_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/5_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/6_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/7_out.png) |
|BMW 1 Series Coupe 2012, prob: 0.9948|Suzuki Aerio Sedan 2007, prob: 0.9982|Ford Mustang Convertible 2007, prob: 1.0|BMW 1 Series Convertible 2012, prob: 1.0|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/8_out.png)  | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/9_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/10_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/11_out.png)|
|Mitsubishi Lancer Sedan 2012, prob: 0.4401|Cadillac CTS-V Sedan 2012, prob: 0.9801|Chevrolet Traverse SUV 2012, prob: 0.9999|Bentley Continental GT Coupe 2012, prob: 0.9953|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/12_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/13_out.png)| ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/14_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/15_out.png)|
|Nissan Juke Hatchback 2012, prob: 0.9935|Chevrolet TrailBlazer SS 2009, prob: 0.987|Hyundai Accent Sedan 2012, prob: 0.9826|Ford Fiesta Sedan 2012, prob: 0.6502|
|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/16_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/17_out.png)|![image](https://github.com/foamliu/Car-Recognition/raw/master/images/18_out.png) | ![image](https://github.com/foamliu/Car-Recognition/raw/master/images/19_out.png)|
|Acura TL Sedan 2012, prob: 0.9999|Aston Martin V8 Vantage Coupe 2012, prob: 0.5487|Infiniti G Coupe IPL 2012, prob: 0.2621|Ford F-150 Regular Cab 2012, prob: 0.9995|
