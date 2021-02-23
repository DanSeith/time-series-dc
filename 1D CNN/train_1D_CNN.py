from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import layers
import os
import numpy as np

# Load dataset
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/Full_filt_101_cx_el.pkl')

feature_list = ['padded_aspect', 'padded_perimeter', 'padded_area', 'padded_deform']

x = df[feature_list].to_numpy()
y = df[['y']].to_numpy()

# Split test and train data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=123)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

# Use function for making specific models, allows model architectures to be recreated with random parameters for
# testing purposes


def create_model():
    _model = tf.keras.models.Sequential()
    _model.add(tf.keras.Input(shape=(None, 35, 4)))
    _model.add(layers.Conv1D(16, 10, activation='relu'))
    _model.add(layers.Conv1D(16, 10, activation='relu'))
    _model.add(layers.Conv1D(16, 10, activation='relu'))
    _model.add(layers.Dense(16, activation='relu'))
    _model.add(layers.Dense(16, activation='relu'))
    _model.add(layers.Dense(1, activation='sigmoid'))
    _model.compile(optimizer='rmsprop',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                   metrics=['accuracy']
                   )
    return _model


model = create_model()
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=1,
                                                 verbose=0)

history = model.fit(x_train, y_train, epochs=350,
                    validation_data=(x_val, y_val),
                    callbacks=[cp_callback])


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('1D_CNN_training.png', dpi=300)

# Create base model and load best validation weights
cp = np.argmax(history.history['val_accuracy'])
val_model = create_model()
model.load_weights('cp-' + str.zfill(str(cp), 4) + '.ckpt')
test_acc = model.evaluate(x_test, y_test)[1]
print(test_acc)

