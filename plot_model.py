from keras.utils import plot_model

from utils import load_model

if __name__ == '__main__':
    model = load_model()
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)
