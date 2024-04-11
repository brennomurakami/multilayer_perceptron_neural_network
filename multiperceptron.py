import openpyxl
import random
import numpy as np


def percent_training():
    percent_train = float(input("Enter percent for training (just numbers):"))
    if percent_train > 1:
        percent_train = percent_train / 100
    return percent_train


def read_excel(file_path, percent_of_training):
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active

    data = []

    # Capturing dat
    for row_index, row in enumerate(ws.iter_rows(values_only=True), start=1):
        # ignoring header
        # Size / Weight / Sweetness / Crunchiness / Juiciness / Ripeness / Acidity
        if row_index == 1:
            continue
        # Adding line data
        data.append(row)

    # calculing training
    total_rows = len(data)
    num_training_rows = int(percent_of_training * total_rows)

    # Shuffle Data
    random.shuffle(data)

    # Separar dados de treinamento e teste
    training_data = data[:num_training_rows]
    testing_data = data[num_training_rows:]

    return training_data, testing_data


# defining percent training
percent_training = percent_training()

# setting data path
filename = "/home/breno/PycharmProjects/multipercepctron/multilayer_perceptron_neural_network/data/apple_quality.xlsx"
train_data, test_data = read_excel(filename, percent_training)

# Convert the lists in numpy arrays
# train_data = np.array(train_data)
# test_data = np.array(test_data)

print("Rows for Training Data: ", len(train_data))
print("Rows for Test Data: ", len(test_data))

print(train_data)
