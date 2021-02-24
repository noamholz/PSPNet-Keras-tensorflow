from os import path
from os.path import join
# from scipy.misc import imresize
from utils.preprocessing import data_generator_s31, preprocess_img
from utils.callbacks import callbacks
from tensorflow.keras.models import load_model
import layers_builder as layers
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def set_npy_weights(weights_path, model):
    npy_weights_path = join("weights", "npy", weights_path + ".npy")
    json_path = join("weights", "keras", weights_path + ".json")
    h5_path = join("weights", "keras", weights_path + ".h5")

    print("Importing weights from %s" % npy_weights_path)
    weights = np.load(npy_weights_path, allow_pickle=True, encoding="latin1").item()

    for layer in model.layers:
        print(layer.name)
        if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
            mean = weights[layer.name]['mean'].reshape(-1)
            variance = weights[layer.name]['variance'].reshape(-1)
            scale = weights[layer.name]['scale'].reshape(-1)
            offset = weights[layer.name]['offset'].reshape(-1)

            model.get_layer(layer.name).set_weights(
                [scale, offset, mean, variance])

        elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
            try:
                weight = weights[layer.name]['weights']
                model.get_layer(layer.name).set_weights([weight])
            except Exception as err:
                try:
                    biases = weights[layer.name]['biases']
                    model.get_layer(layer.name).set_weights([weight,
                                                             biases])
                except Exception as err2:
                    print(err2)

        if layer.name == 'activation_52':
            break


def train(datadir, logdir, input_size, nb_classes, resnet_layers, batchsize, weights, initial_epoch, pre_trained, sep):
    if args.weights:
        model = load_model(weights)
    else:
        model = layers.build_pspnet(nb_classes=nb_classes,
                                    resnet_layers=resnet_layers,
                                    input_shape=input_size)
        set_npy_weights(pre_trained, model)
    dataset_len = len(os.listdir(os.path.join(datadir, 'img')))
    train_generator, val_generator = data_generator_s31(
        datadir=datadir, batch_size=batchsize, input_size=input_size, nb_classes=nb_classes, separator=sep)
    model.save('weights_train/pretrained_ade20k_473x713.h5')
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        validation_steps=20,
        epochs=2, verbose=True, steps_per_epoch=1,
        callbacks=callbacks(logdir), initial_epoch=initial_epoch)


def predict(datadir, logdir, input_size, nb_classes, resnet_layers, batchsize, weights, initial_epoch, pre_trained, sep):
    if args.weights:
        model = load_model(weights)
    else:
        model = layers.build_pspnet(nb_classes=nb_classes,
                                    resnet_layers=resnet_layers,
                                    input_shape=input_size)
        if False:
            model.load_weights('weights_train/weights.01-0.05.h5')
        else:
            set_npy_weights(pre_trained, model)
        # set_npy_weights(pre_trained, model)
    train_generator, val_generator = data_generator_s31(
        datadir=datadir, batch_size=batchsize, input_size=input_size, nb_classes=nb_classes, separator=sep)
    DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])
    img = np.array(next(iter(val_generator))[0])[0, ..., ::-1]
    img = img - DATA_MEAN
    img = img[:, :, ::-1]
    img.astype('float32')
    a = model.predict_on_batch(img[None, ...])
    plt.imshow(batch[0][0, ...].astype('uint8')[..., ::-1])
    plt.imshow(a[0, ..., 1]>0.5, alpha=0.4)
    plt.show()
    plt.imshow(model.predict_on_batch(batch - DATA_MEAN[None, ..., ::-1]).argmax(axis=-1)[0,...])
    plt.show()


class PSPNet(object):
    """Pyramid Scene Parsing Network by Hengshuang Zhao et al 2017"""

    def __init__(self, nb_classes, resnet_layers, input_shape):
        self.input_shape = input_shape
        self.model = layers.build_pspnet(nb_classes=nb_classes,
                                         layers=resnet_layers,
                                         input_shape=self.input_shape)
        print("Load pre-trained weights")
        self.model.load_weights("weights/keras/pspnet101_cityscapes.h5")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', nargs='+', type=int, default=(473, 713))
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--resnet_layers', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--datadir', type=str, required=False, default='../../../../../datasets/annots_findgrass/rgo_annots_20201023_color/20200820_Ofek10b_color/')
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='pspnet50_ade20k',
                        help='Model/Weights to use',
                        choices=['pspnet50_ade20k',
                                 'pspnet101_cityscapes',
                                 'pspnet101_voc2012'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sep', default=').')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # predict('../../../../../datasets/annots_findgrass/rgo_annots_20201023_color/20200910_Ofek_ShabtaiArea/', args.logdir, (473, 473), args.classes, args.resnet_layers,
    #       args.batch, args.weights, args.initial_epoch, args.model, args.sep)
    train(args.datadir, args.logdir, args.input_dim, args.classes, args.resnet_layers,
          args.batch, args.weights, args.initial_epoch, args.model, args.sep)
