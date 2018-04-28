#from resnet_50 import resnet50_model
from resnet_152 import resnet152_model
from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH, IMG_HEIGHT = 227, 227
TRAIN_DATA = 'mart/standford-cars-crop/train'
VALID_DATA = 'mart/standford-cars-crop/valid'
NUM_CLASSES = 196
NB_TRAIN_SAMPLES = 6549
NB_VALID_SAMPLES = 1595
BATCH_SIZE = 16

# build a classifier model
#model = resnet50_model(IMG_HEIGHT, IMG_WIDTH, 3, NUM_CLASSES)
model = resnet152_model(IMG_HEIGHT, IMG_WIDTH, 3, NUM_CLASSES)

# prepare data augmentation configuration
train_data_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.2, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
valid_data_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.2, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

train_generator = train_data_gen.flow_from_directory(TRAIN_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')
valid_generator = valid_data_gen.flow_from_directory(VALID_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')

# fine tune the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=NB_VALID_SAMPLES // BATCH_SIZE,
    epochs=80)

model.save_weights("model.h5")
