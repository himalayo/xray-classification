#!/usr/bin/env python3
import keras
import tensorflow as tf
from keras.optimizers import SGD
from keras.applications.efficientnet_v2 import preprocess_input
from keras.applications import ConvNeXtTiny
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, RandomFlip, RandomContrast, RandomBrightness
import json
import argparse

available_models = {
        "convnext": keras.applications.ConvNeXtTiny,
        "vgg16": keras.applications.VGG16,
        "resnet50": keras.applications.ResNet50,
        "mobilenet": keras.applications.MobileNetV2,
        "efficientnet": keras.applications.EfficientNetB0,
        "inception": keras.applications.InceptionV3
        }

preprocessing_function = {
        "vgg16": keras.applications.vgg16.preprocess_input,
        "resnet50": keras.applications.resnet.preprocess_input,
        "mobilenet": keras.applications.mobilenet.preprocess_input,
        "inception": keras.applications.inception_v3.preprocess_input
        }

def create_model(model_name, models = available_models, dense_units = 512, learning_rate = 0.00001):
    base_model = models[model_name.lower()](weights = 'imagenet', include_top = False)
    base_model.trainable = False
    for layer in base_model.layers[-(len(base_model.layers)//10):]:
        if not isinstance(layer, keras.layers.LayerNormalization):
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation = 'relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = Flatten()(x)
    prediction_layer =  Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = prediction_layer)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate),
            loss = 'crossentropy',
            metrics = [keras.metrics.BinaryAccuracy(),
                       keras.metrics.Precision(),
                       keras.metrics.Recall(),
                       keras.metrics.F1Score(),
                       keras.metrics.FalseNegatives(),
                       keras.metrics.FalsePositives()
                       ])

    model.summary()
    return model


def load_data(model_name, preprocess_fns = preprocessing_function):
    training_data = keras.utils.image_dataset_from_directory(
            directory = './chest_xray/train',
            label_mode = 'binary',
            image_size = (799,352),
            )

    vali_data = test_data = keras.utils.image_dataset_from_directory(
            directory = './chest_xray/test',
            label_mode = 'binary',
            image_size = (799,352)
            )
    model_name = model_name.lower()

    if model_name in preprocess_fns:
        fn = preprocess_fns[model_name]
        training_data = training_data.map(lambda xs, ys: (fn(xs), ys))
        vali_data = vali_data.map(lambda xs, ys: (fn(xs), ys))
        test_data = test_data.map(lambda xs, ys: (fn(xs), ys))

    return training_data, vali_data, test_data

def handle_arguments():
    parser = argparse.ArgumentParser(prog="XRay-Training",
                                     description="Trains different models on the chest XRay dataset")
    parser.add_argument("-m", "--model", help="Model to use", choices=["convnext", "vgg16", "resnet50", "mobilenet", "efficientnet", "inception"], type=str.lower, default="convnext")
    parser.add_argument("-d", "--dense_units", type=int, help="Density units for the classification head. Defaults to 512", default=512)
    parser.add_argument("-l", "--learning_rate", type=float, help="Learning rate to be used during training", default=0.00001)
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs", default=10)
    args = parser.parse_args()
    return args.model, args.dense_units, args.learning_rate, args.epochs

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=3572)])

    model_name, dense_units, lr, epochs = handle_arguments()
    training_data, vali_data, test_data = load_data(model_name)
    model = create_model(model_name, dense_units = dense_units, learning_rate = lr)


    model.fit(training_data, 
            epochs = epochs,
              validation_data = vali_data,
            callbacks = [
                keras.callbacks.ModelCheckpoint(filepath=f"./tmp_{model_name}_{dense_units}/chck/{{epoch:02d}}_{{binary_accuracy:.8f}}.keras", monitor='binary_accuracy', save_freq=100),
                keras.callbacks.BackupAndRestore(backup_dir=f"./tmp_{model_name}_{dense_units}/backups", save_freq=100),
                keras.callbacks.CSVLogger(f'./{model_name}_{dense_units}.log')
                ])

    model.trainable = True
    model.save(f"{model_name}_{dense_units}.keras")
    model = keras.models.load_model(f"{model_name}_{dense_units}.keras")

    model.evaluate(test_data)
