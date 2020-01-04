import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_handle import load_data, load_batches


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def sigmoid_gradient(Z):
    return sigmoid(Z) * (1.0 - sigmoid(Z))


def relu(Z):
    return np.maximum(0.0, Z)


def relu_gradient(Z):
    temp = np.array(Z, copy=True)
    temp[temp <= 0] = 0.0
    temp[temp > 0] = 1.0
    return temp


def softmax(z):
    max_row = np.max(z, axis=-1, keepdims=True)  # 每一个样本的所有分数中的最大值
    tmp = z - max_row
    return np.exp(tmp) / np.sum(np.exp(tmp), axis=-1, keepdims=True)


def softmax_cross_entropy(logits, y):
    n = logits.shape[0]
    a = softmax(logits)
    scores = a[range(n), y]
    # scores = a[range(n), np.argmax(y, axis=1)]
    loss = -np.sum(np.log(scores)) / n
    return a, loss


def derivation_softmax_cross_entropy(logits, y):
    n = logits.shape[0]
    a = softmax(logits)
    a[range(n), y] -= 1
    # a[range(n), np.argmax(y, axis=1)] -= 1
    return a


class Layer:
    def __init__(self, layers_dims, activation='relu'):
        """
        构造函数
        :param layers_dims: (input_size, output_size) 权值矩阵的大小（上一层的神经元个数，本层的神经元个数）
        :param activation:  激活函数
        """
        self.W = 2.0 * np.random.random(layers_dims) - 1.0
        self.b = 0.0
        self.act = activation
        self.W_gradient = 1.0e-10
        self.b_gradient = 1.0e-10
        self.err = np.array([])
        self.gradiant = np.array([])
        self._input = np.array([])
        self._output = np.array([])

    def _forward(self, X, Y):
        """
        神经层的前向计算
        :param X: 上一层的输出
        :param Y: 样本的标签，用于最后一层的计算。
        :return:
        """
        self._input = np.dot(X, self.W) + self.b
        loss = 0
        if self.act == 'relu':
            self._output = relu(self._input)
            loss = self._output
        elif self.act == 'sigmoid':
            self._output = sigmoid(self._input)
            loss = self._output
        elif self.act == 'softmax':
            a, loss = softmax_cross_entropy(self._input, Y)
            self._output = a
        elif self.act == 'linear':
            self._output = self._input
            loss = self._output
        return self._output, loss

    def count_err(self, front_err, Y):
        """
        计算本层的误差
        :param front_err: 后一层的误差
        :param Y: 样本的标签，用于最后一层的计算。
        :return:
        """
        if self.act == 'softmax':
            self.err = derivation_softmax_cross_entropy(self._input, Y)
        else:
            self.err = front_err * self.act_gradient(self._input)
        return self.err

    def count_gradient(self, al_1):
        """

        :param al_1: 前一层的a
        :return:
        """
        self.gradiant = np.dot(al_1.T, self.err)

    def update_parameters(self, rate, n):
        """

        :param rate: 学习率
        :param n: batch_size
        :return:
        """
        self.W = self.W - rate * self.gradiant / n
        self.b = self.b - rate * np.sum(self.err) / n

    def act_gradient(self, Z):
        """

        :param Z: 本层的Z
        :return:
        """
        if self.act == 'relu':
            return relu_gradient(Z)
        elif self.act == 'sigmoid':
            return sigmoid_gradient(Z)
        elif self.act == 'linear':
            return 1


class NeuralNetwork:

    def __init__(self):
        self.NN = []

    def add_layer(self, layers_dims, activation='relu'):
        """

        :param layers_dims: (input_size, output_size) 权值矩阵的大小（上一层的神经元个数，本层的神经元个数）
        :param activation: 激活函数
        :return:
        """
        self.NN.append(Layer(layers_dims, activation))

    def forward(self, X, Y, mission):
        """

        :param X: input
        :param Y: output
        :param mission: 分类还是拟合 regression | classification
        :return:
        """
        temp = X
        loss = 0.0
        for nn in self.NN:
            temp, loss = nn._forward(temp, Y)
        if mission == 'regression':
            temp = self.NN[-1]._input
        return temp, loss

    def backward(self, out, x, y, rate, mission):
        """
        反向传播
        :param out: 拟合时的输出
        :param x: 样本特征
        :param y: 样本标签
        :param rate: 学习率
        :param mission: 分类还是拟合 regression | classification
        :return:
        """
        self.NN.reverse()
        _err = 0.0
        if mission == 'regression':
            _err = 2 * (out - y)
        for i in self.NN:
            _err = i.count_err(_err, y)
            _err = np.dot(_err, i.W.T)
        if len(self.NN) > 1:
            for i in list(range(len(self.NN)))[0:-1]:
                self.NN[i].count_gradient(self.NN[i + 1]._output)
        self.NN[-1].count_gradient(x)
        self.NN.reverse()
        for i in self.NN:
            i.update_parameters(rate, y.shape[0])

    def train(self, x, y, mission, epochs=1000, batch_size=50, learning_rate=0.01, data_rate=0.6, target=None, print_cost=False):
        j = 0
        x_train, y_train, x_val, y_val = load_data(x, y, data_rate)
        cost_his = []
        while j < epochs:
            # 训练
            x_batches, y_batches = load_batches(x_train, y_train, batch_size)
            for i, (temp_x, temp_y) in enumerate(zip(x_batches, y_batches)):
                out, cost = self.forward(temp_x, temp_y, mission)
                self.backward(out, temp_x, temp_y, learning_rate, mission)

            # 验证
            j += 1
            if print_cost and j % 10 == 0:
                x_batches, y_batches = load_batches(x_val, y_val, batch_size)
                costs = 0
                acc = 0.0
                n = x_val.shape[0]
                for i, (temp_x, temp_y) in enumerate(zip(x_batches, y_batches)):
                    out, cost = self.forward(temp_x, temp_y, mission)
                    if mission == 'regression':
                        costs += np.sum((temp_y - out) ** 2)
                    elif mission == 'classification':
                        logits = self.NN[-1]._input
                        correct = np.sum(np.argmax(logits, axis=-1) == temp_y)
                        acc += correct
                        costs += cost
                costs /= n
                cost_his.append(costs)
                acc /= n
                if mission == 'classification':
                    print("epochs %i cost value: %f, acc value: %f" % (j, costs, acc))
                elif mission == 'regression':
                    print("epochs %i cost value: %f" % (j, costs))

                if target != None:
                    if mission == 'regression':
                        if float(costs) < target:
                            break
                        elif j == epochs:
                            j = 0
                            learning_rate *= 0.95
                    elif mission == 'classification':
                        if float(acc) > target:
                            break
                        elif j == epochs:
                            j = 0
                            learning_rate *= 0.95

    def predict(self, x, y, mission):
        if mission == 'regression':
            temp = x
            for nn in self.NN:
                temp, loss = nn._forward(temp, y)
            return temp
        elif mission == 'classification':
            temp = x
            for nn in self.NN:
                temp, loss = nn._forward(temp, y)
            logits = self.NN[-1]._input
            return np.argmax(logits, axis=-1) + 1
            # correct = np.sum(np.argmax(logits, axis=-1) == y)
            # acc = correct / y.shape[0]
            # print('acc: ', acc)

    def save_model(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, f_d):
        with open(f_d, 'rb') as f:
            a = pickle.load(f)
            self.NN = a.NN


if __name__ == '__main__':
    a = NeuralNetwork()
    a.add_layer([1, 24], activation='sigmoid')
    a.add_layer([24, 10], activation='sigmoid')
    a.add_layer([10, 1], activation='linear')
    a.load_model('test.pkl')
    print(a.NN)


