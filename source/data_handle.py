import numpy as np
import random


def load_data(x, y, rate):
    if rate == 1.0:
        return x, y, x, y
    else:
        n = int(x.shape[0] * rate)
        return x[0:n], y[0:n], x[n:], y[n:]


def load_batches(x, y, batch_size):
    n = x.shape[0]
    # 对数据进行洗牌
    shuffle_idx = random.sample(range(n), n)
    X = x[shuffle_idx]
    Y = y[shuffle_idx]

    batches_x = [X[i: i + batch_size] for i in range(0, n, batch_size)]
    batches_y = [Y[i: i + batch_size] for i in range(0, n, batch_size)]

    return batches_x, batches_y


if __name__ == '__main__':
    import cv2
    import os
    import numpy as np
    from nn import NeuralNetwork
    # X_data = -np.pi + (np.pi + np.pi) * np.random.rand(10000, 1)
    # Y_data = np.sin(X_data)
    # x_train, y_train, x_val, y_val = load_data(X_data, Y_data, 0.6, 'regression')
    # x_batches, y_batches = load_batches(x_val, y_val, 50)
    # print([x.shape for x in x_batches])

    root_dir = r'F:\WangHao\Money\NN\LAB1\train'
    data = np.zeros((12 * 620, 28 * 28 + 1))
    for i in range(12):
        for j in range(620):
            temp_data = cv2.imread(os.path.join(root_dir, str(i + 1), str(j + 1) + '.bmp'), cv2.IMREAD_GRAYSCALE)
            data[i * 620 + j, 0:28 * 28] = np.reshape(temp_data, (28 * 28))
            data[i * 620 + j, -1] = i + 1

    X = data[:, 0:28 * 28] / 255.0
    Y_temp = data[:, -1]
    Y = Y_temp.astype(dtype=np.int) - 1
    print(X.shape, Y.shape)
    x_train, y_train, x_val, y_val = load_data(X, Y, 0.5, 'classification')
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    x_batches, y_batches = load_batches(x_val, y_val, 50)
    print([x.shape for x in x_batches])
