import openpyxl
import numpy as np


def read_excel(filename, percent_training):
    wb = openpyxl.load_workbook(filename)
    ws = wb.active

    data = []

    for row in ws.iter_rows(values_only=True):
        # read the values of the first 8 columns
        row_data = row[:7]
        data.append(row_data)

    return data


percent_training = float(input("Enter percent for training (just numbers):"))
if percent_training > 1:
    percent_training = percent_training / 100

print(percent_training)

filename = "/home/breno/PycharmProjects/multipercepctron/apple_quality.xlsx"
# Size / Weight / Sweetness / Crunchiness / Juiciness / Ripeness / Acidity
data = read_excel(filename, percent_training)
print(data)