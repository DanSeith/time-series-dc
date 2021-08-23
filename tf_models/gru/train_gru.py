from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import sklearn
import pandas as pd
import sys
# Import df_utils
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
import df_utils
from sklearn.preprocessing import LabelBinarizer

# Load dataset
df = pd.read_pickle(os.path.join(ROOT_DIR, 'FINAL_DF_light'))

x = np.load(os.path.join(ROOT_DIR, 'x_all.npy'))
y = np.load(os.path.join(ROOT_DIR, 'y_all.npy'))
y = LabelBinarizer().fit_transform(y)

# Split test and train data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=123)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

# Use function for making specific models, allows model architectures to be recreated with random parameters for
# testing purposes
# ****** Experiments *****
# Dropout - no dropout is best
# Dense layers - square 1 is best - triangle
# Dense units
# Learning rate
# Optimizer
#
tf.random.set_seed(123)


def create_model(loss, units, dense_units):
    if units > 1:
        activation = 'softmax'
    else:
        activation = 'sigmoid'
    _model = tf.keras.models.Sequential()
    _model.add(layers.Masking(input_shape=x_train.shape[1:]))
    _model.add(layers.GRU(x_train.shape[1], return_sequences=True))
    _model.add(layers.GRU(x_train.shape[1]))
    for du in range(dense_units):
        _model.add(layers.Dense(int(24/(2**du)), activation='relu'))
    _model.add(layers.Dense(units, activation=activation))
    _model.compile(optimizer='adam',
                   loss=loss,
                   metrics=['accuracy']
                   )
    return _model


if y_val[0].shape[0] > 2:
    model_loss = tf.keras.losses.CategoricalCrossentropy()
    units = y_val[0].shape[0]
else:
    model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
    units = 1

for d_units in [2, 3]:
    model = create_model(model_loss, units, d_units)
    base_path = os.getcwd()
    # shutil.rmtree('training_1/')
    checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_freq=1,
                                                     verbose=0)

    history = model.fit(x_train, y_train, epochs=100,
                        validation_data=(x_val, y_val),
                        callbacks=[cp_callback],
                        verbose=0)
    print(np.max(history.history['val_accuracy']))
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.savefig('gru_training.eps', format='eps')
# plt.figure()
# os.chdir(os.path.join(base_path, 'training_1'))
#
# # Create base model and load best validation weights
# cp = np.argmax(history.history['val_accuracy'])
# val_model = create_model(model_loss, units)
# model.load_weights('cp-' + str.zfill(str(cp), 4) + '.ckpt')
# test_acc = model.evaluate(x_test, y_test)[1]
# print(test_acc)
#
# os.chdir(base_path)
# pred = model.predict(x_test).ravel()
# fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, pred)
# auc = np.expand_dims(sklearn.metrics.auc(fpr, tpr), -1)
#
# # Save tpr and fpr for lstm model comparison script
# np.savetxt('gru_fpr.csv', fpr)
# np.savetxt('gru_tpr.csv', tpr)
# np.savetxt('gru_auc.csv', auc)
#
# # Save npy of y_pred/y_true
# np.save('gru_pred.npy', np.vstack((pred, y_test)))
#
# plt.plot(fpr, tpr, label='GRU' + ' (AUC: ' + str(round(float(auc), 2)) + ')')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC Curve')
# plt.legend(loc='best')
# plt.savefig('gru_roc.png', dpi=300)
