import keras
import tensorflow as tf
from keras.optimizers import SGD
from keras.applications.efficientnet_v2 import preprocess_input
from keras.applications import ConvNeXtTiny
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, RandomFlip, RandomContrast, RandomBrightness
import json

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
physical_devices[0],
[tf.config.LogicalDeviceConfiguration(memory_limit=3572)])

print('Loading datasets...')
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


#training_data = training_data.map(lambda xs,ys: (training_preprocess(xs),ys))
#test_data = test_data.map(lambda xs,ys: (preprocess_input(xs),ys))

print('Setting up model...')
base_model = ConvNeXtTiny(weights = 'imagenet', include_top = False)
base_model.trainable = False
for layer in base_model.layers[-(len(base_model.layers)//10):]:
    if not isinstance(layer, keras.layers.LayerNormalization):
        layer.trainable = True
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation = 'relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = Flatten()(x)
prediction_layer =  Dense(1, activation = 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = prediction_layer)
#base_model.trainable = False
model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.00001), 
        loss = 'crossentropy',
        metrics = [keras.metrics.BinaryAccuracy(),
                   keras.metrics.Precision(),
                   keras.metrics.Recall(),
                   keras.metrics.F1Score(),
                   keras.metrics.FalseNegatives(),
                   keras.metrics.FalsePositives()
                   ])
model.summary()

print('Initial training...')

model.fit(training_data, 
        epochs = 10,
          validation_data = vali_data,
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath='./tmp_convnexttiny/chck/{epoch:02d}_{binary_accuracy:.8f}.keras', monitor='binary_accuracy', save_freq=100),
            keras.callbacks.BackupAndRestore(backup_dir='./tmp_convnexttiny/backups', save_freq=100),
            keras.callbacks.CSVLogger('./convnexttiny.log')
            ])

model.trainable = True
model.save('initial_convnexttiny.keras')
model = keras.models.load_model('initial_convnexttiny.keras')

#print('Fine-tuning...')
#model.compile(optimizer=SGD(learning_rate = 0.0001, momentum = 0.9), 
#              loss='crossentropy',
#        metrics = [keras.metrics.BinaryAccuracy(),
#                   keras.metrics.Precision(),
#                   keras.metrics.Recall(),
#                   keras.metrics.F1Score(),
#                   keras.metrics.FalseNegatives(),
#                   keras.metrics.FalsePositives()
#                   ])
#model.fit(training_data, epochs = 2,
#          validation_data = vali_data,
#            callbacks=[ keras.callbacks.CSVLogger('./convnexttiny.log', append=True),
#                        keras.callbacks.ModelCheckpoint(filepath='./tmp_convnexttiny/chck_tuning/{epoch:02d}_{binary_accuracy:.8f}.keras', monitor='binary_accuracy', save_freq=100),
#                        keras.callbacks.BackupAndRestore(backup_dir='./tmp_convnexttiny/backups_tuning', save_freq=100)])
#model.save('convnexttiny_ray.keras')
model.evaluate(test_data)
