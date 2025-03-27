import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow import keras

# Reading the Dataset
path = "/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv"
red_wine  = pd.read_csv(path)
red_wine .head()
# splitting the dataset
df_train = red_wine.sample(frac=0.7)
df_valid = red_wine.drop(df_train.index)

# Normalalization
max = df_train.max(axis=0)
min = df_train.min(axis=0)

df_train = (df_train - min) / (max - min)
df_valid = (df_valid - min) / (max - min)

# Defining features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']
X_train.shape[1]
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[11]),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1)
])

model.compile(
    optimizer = "adam",
    loss = "mae"
)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=1,  # turn off training log
)