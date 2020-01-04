import cv2
import os
import numpy as np
from nn import NeuralNetwork


def train_classification():
    root_dir = r'..\LAB1\train'
    data = np.zeros((12*620, 28*28+1))
    for i in range(12):
        for j in range(620):
            temp_data = cv2.imread(os.path.join(root_dir, str(i+1), str(j+1)+'.bmp'), cv2.IMREAD_GRAYSCALE)
            data[i*620+j, 0:28*28] = np.reshape(temp_data, (28*28))
            data[i*620+j, -1] = i + 1

    a = NeuralNetwork()
    a.add_layer([28 * 28, 10*10])
    # a.add_layer([14 * 14, 12 * 12])
    a.add_layer([10*10, 12], activation='softmax')
    X = data[:, 0:28*28]/255.0
    Y_temp = data[:, -1]
    Y = Y_temp.astype(dtype=np.int) - 1
    # Y = np.zeros((Y_temp.shape[0], 12))
    # for i in range(Y_temp.shape[0]):
    #     Y[i, int(Y_temp[i]-1)] = 1
    a.train(X, Y, 'classification', learning_rate=0.01, batch_size=50, data_rate=1.0, target=0.9, print_cost=True)
    a.save_model('hand_writing.pkl')


def predict_classification():
    a = NeuralNetwork()
    a.load_model('hand_writing.pkl')
    root_dir = r'..\LAB1\train'
    file_class = 10
    file_num = '19'
    file_dir = os.path.join(root_dir, str(file_class), file_num + '.bmp')
    data = np.zeros((1, 28 * 28))
    temp_data = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
    data[0, :] = np.reshape(temp_data, (28 * 28))
    x = data / 255.0
    y = np.array([file_class])
    y_predict = a.predict(x, y, 'classification')
    print(y_predict)


train_classification()
# predict_classification()