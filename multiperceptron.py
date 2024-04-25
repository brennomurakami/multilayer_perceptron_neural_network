import openpyxl
import random
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
# Preset Matplotlib figure sizes.
matplotlib.rcParams['figure.figsize'] = [9, 6]
import tensorflow as tf


def define_percent_training():
    percent_train = float(input("Enter percent for training (just numbers):"))
    if percent_train > 1:
        percent_train = percent_train / 100
    return percent_train


def read_excel(file_path, percent_of_training):
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active

    data = []

    # Capturing all data
    for row_index, row in enumerate(ws.iter_rows(values_only=True), start=1):
        # ignoring header
        if row_index == 1:
            continue
        # Adding line data
        data_row = list(row)
        # Convertendo 'good' para 1 e 'bad' para 0 na coluna 8
        if data_row[-1] == 'good':
            data_row[-1] = 1
        else:
            data_row[-1] = 0
        # Convertendo os valores para floats
        data_row = [float(value) if isinstance(value, (int, float)) else value for value in data_row]
        data.append(data_row)

    # calculating training
    total_rows = len(data)
    num_training_rows = int(percent_of_training * total_rows)

    # Shuffle Data
    random.shuffle(data)

    # Separar dados de treinamento e teste
    training_data = data[:num_training_rows]
    testing_data = data[num_training_rows:]

    return training_data, testing_data


def build_model(input_size, hidden_layer_sizes, output_size):
    model = tf.keras.Sequential()
    for size in hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))
    model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))
    return model


def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    return history


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)


# defining percent training
percent_training = define_percent_training()

# setting data path
filename = "data/apple_quality.xlsx"
train_data, test_data = read_excel(filename, percent_training)

# Convert the lists in numpy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)

# Convert the lists in numpy arrays
x_train = np.array([data[:-1] for data in train_data])
y_train = np.array([data[-1] for data in train_data])
x_test = np.array([data[:-1] for data in test_data])
y_test = np.array([data[-1] for data in test_data])

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

print("x_train")
for i in x_train:
    print(i.dtype)

print("y_train")
for i in y_train:
    print(i.dtype)

print("x_test")
for i in x_test:
    print(i.dtype)

print("y_test")
for i in y_test:
    print(i.dtype)

print("Rows for Training Data: ", len(train_data))
print("Rows for Test Data: ", len(test_data))

# Definição dos parâmetros da rede neural
input_size = len(x_train[0])  # Tamanho da entrada
hidden_layer_sizes = [64, 32]  # Número de neurônios em cada camada oculta
output_size = 1               # Tamanho da saída (problema de classificação binária)

# 1. Construir o Modelo
mlp_model = build_model(input_size, hidden_layer_sizes, output_size)

# 2. Compilar o Modelo
compile_model(mlp_model)

# 3. Treinar o Modelo
history = train_model(mlp_model, x_train, y_train, x_test, y_test)

# 4. Avaliar o Modelo
evaluate_model(mlp_model, x_test, y_test)


