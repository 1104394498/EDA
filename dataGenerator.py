import random
import numpy as np
import json

datasets_num = 10 ** 4
datasets_train = []
datasets_test = []
train_ratio = 0.7

input_x = 20
input_y = 20

for i in range(datasets_num):
    dim0_mat = []
    dim1_mat = []
    for _ in range(input_x):
        r = random.random()
        dim0_mat.append([r for _ in range(input_y)])
    for _ in range(input_y):
        r = random.random()
        dim1_mat.append([r for _ in range(input_x)])
    # transpose dim1_mat
    dim1_mat = np.array(dim1_mat)
    dim1_mat = dim1_mat.T.tolist()

    pin_mat = np.zeros((input_x, input_y))

    pin_num = random.randint(2, min(1000, input_x * input_y // 3))
    pin_locs = []
    for _ in range(pin_num):
        while True:
            pin_x = random.randint(0, input_x - 1)
            pin_y = random.randint(0, input_y - 1)
            if (pin_x, pin_y) not in pin_locs:
                pin_locs.append((pin_x, pin_y))
                pin_mat[pin_x, pin_y] = 1
                break

    pin_mat = pin_mat.tolist()

    matrix = [dim0_mat, dim1_mat, pin_mat]
    curDict = dict()
    curDict['matrix'] = matrix
    curDict['pin_locs'] = pin_locs

    if i < datasets_num * train_ratio:
        datasets_train.append(curDict)
    else:
        datasets_test.append(curDict)

    if i % 10 == 0:
        print('NO.%d finish' % i)

with open(r'try_datasets_train.json', 'w') as f:
    json.dump(datasets_train, f)
with open(r'try_datasets_test.json', 'w') as f:
    json.dump(datasets_test, f)