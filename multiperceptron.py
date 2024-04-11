import openpyxl
import random
import numpy as np


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
        data.append(list(row))

    # calculating training
    total_rows = len(data)
    num_training_rows = int(percent_of_training * total_rows)

    # Shuffle Data
    random.shuffle(data)

    # Separar dados de treinamento e teste
    training_data = data[:num_training_rows]
    testing_data = data[num_training_rows:]

    # Modificar os dados na última coluna
    for i in range(len(training_data)):
        if training_data[i][-1] == "good":
            training_data[i][-1] = 1
        else:
            training_data[i][-1] = 0

    for i in range(len(testing_data)):
        if testing_data[i][-1] == "good":
            testing_data[i][-1] = 1
        else:
            testing_data[i][-1] = 0

    return training_data, testing_data


# defining percent training
percent_training = define_percent_training()

# setting data path
filename = "/home/breno/PycharmProjects/multipercepctron/multilayer_perceptron_neural_network/data/apple_quality.xlsx"
train_data, test_data = read_excel(filename, percent_training)

# Convert the lists in numpy arrays
# train_data = np.array(train_data)
# test_data = np.array(test_data)

print("Rows for Training Data: ", len(train_data))
print("Rows for Test Data: ", len(test_data))

print(train_data)

