from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


def train_sin():
    a = NeuralNetwork()
    a.add_layer([1, 24], activation='sigmoid')
    a.add_layer([24, 10], activation='sigmoid')
    a.add_layer([10, 1], activation='linear')
    X_data = -np.pi + (np.pi + np.pi) * np.random.rand(10000, 1)
    Y_data = np.sin(X_data)
    # plt.scatter(X_data, Y_data)
    # plt.show()

    a.train(X_data, Y_data, 'regression', learning_rate=0.1, target=0.001, print_cost=True)
    a.save_model('sin.pkl')


def predict_sin():
    a = NeuralNetwork()
    a.load_model('sin.pkl')

    X_data = -np.pi + (np.pi + np.pi) * np.random.rand(2000, 1)
    Y_data = np.sin(X_data)
    Y_predict = a.predict(X_data, Y_data, 'regression')
    print(np.sum((Y_predict - Y_data) ** 2)/2000)
    plt.scatter(X_data, Y_predict)
    plt.scatter(X_data, Y_data)
    plt.show()

# 训练数据
train_sin()

# 预测
# predict_sin()