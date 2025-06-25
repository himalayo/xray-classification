#!/usr/bin/env python3

import keras
import numpy as np
import tensorflow as tf
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
physical_devices[0],
[tf.config.LogicalDeviceConfiguration(memory_limit=3572)])
test_data = keras.utils.image_dataset_from_directory(
        directory = './chest_xray/test',
        label_mode = 'binary',
        image_size = (799,352)
        )

preprocessing_function = {
        "vgg16": keras.applications.vgg16.preprocess_input,
        "resnet50": keras.applications.resnet.preprocess_input,
        "mobilenet": keras.applications.mobilenet.preprocess_input,
        "inception": keras.applications.inception_v3.preprocess_input
        }

for filename in sys.argv[1:]:
    model = keras.models.load_model(filename);
    model_name = filename.split(".")[0].split("_")[0]
    data = test_data
    if model_name in preprocessing_function:
        fn = preprocessing_function[model_name]
        data = test_data.map(lambda xs, ys: (fn(xs), ys))
    ys = data.map(lambda xs, ys: ys)
    xs = data.map(lambda xs, ys: xs)
    cm = confusion_matrix(np.concatenate(list(ys.as_numpy_iterator())).squeeze(), np.round(model.predict(xs).squeeze()).tolist())
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot()
    plt.savefig(f"output/{model_name}_confusion.png")
