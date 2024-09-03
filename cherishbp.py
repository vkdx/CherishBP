import numpy as np
import pickle

# ReLU 激活函数
class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)

    def der(self, x):
        t = x.copy()
        t[t > 0] = 1
        return t


# Sigmoid 激活函数
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def der(self, x):
        return x * (1 - x)


# MSE 损失函数
class MSE:
    def __call__(self, true, pred):
        return np.mean(np.power(pred - true, 2), keepdims=True)

    def der(self, true, pred):
        return pred - true


# 定义层
class BPLayer:
    def __init__(self, ins, outs, act_fun, init_mode):
        '''
        ins: 输入神经元个数
        outs: 输出神经元个数
        act_fun: 激活函数
        init_mode: 权重和偏置初始化方式
        '''
        self.ins = ins
        self.outs = outs
        # 初始化权重和偏置，需要考虑网络层数引起的梯度消失和梯度爆炸的问题(init_mode == 1时MSRA初始化方式,否则使用Xavier初始化方式)
        self.weights = np.random.rand(ins, outs) / (np.sqrt(outs) if init_mode == 1 else np.sqrt(2 / outs))

        self.bias = np.random.rand(outs) / (np.sqrt(outs) if init_mode == 1 else np.sqrt(2 / outs))

        self.activation = act_fun
        # 反向传播输入零时值
        self.ins_temp = None
        # 每层经过激活函数的输出结果
        self.act_res = None

    #前向计算
    def __call__(self, ins, BPnet):
        '''
        :param ins: 网络层的输入
        :param BPnet: 网络层容器
        :param mode: ‘train’:训练模型，‘predic’:使用模型进行预测
        :return: 当前层激活计算结果
        '''

        self.ins_temp = ins
        self.act_res = self.activation(np.dot(ins, self.weights) + self.bias)
        #将层添加到BPnet容器中
        if self not in BPnet.layers:
            BPnet.layers.append(self)
            # print(self.__dict__)
        return self.act_res

    #更新权重和偏置
    def update(self, grad, lr):
        activation_diff_grad = self.activation.der(self.act_res) * grad
        # 这个变量肩负重任，将后面的梯度不断向前传播
        new_grad = np.dot(activation_diff_grad, self.weights.T)
        # 参数的更新
        self.weights -= np.dot(lr * self.ins_temp.T, activation_diff_grad)
        # 这里的 mean(axis=0) 与批大小的计算有关
        self.bias -= lr * activation_diff_grad.mean(axis=0)
        # 这里将误差继续往前传
        return new_grad

class CherishBP:
    def __init__(self):
        self.lr = 0.3
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.mse = MSE()

        #用于存储BP层，相当于Sequential
        self.layers = []
        self.BPLayer_1 = BPLayer(1, 14, act_fun=self.relu, init_mode=1)
        self.BPLayer_2 = BPLayer(14, 8, act_fun=self.relu, init_mode=1)
        self.BPLayer_3 = BPLayer(8, 5, act_fun=self.relu, init_mode=1)
        self.BPLayer_4 = BPLayer(5, 1, act_fun=self.sigmoid, init_mode=2)

    # 模型计算
    def __call__(self, ins):
        '''
        :param ins: BP网络的输入
        :return: 网络输出
        '''
        outs = self.BPLayer_1(ins, self)
        outs = self.BPLayer_2(outs, self)
        outs = self.BPLayer_3(outs, self)
        outs = self.BPLayer_4(outs, self)

        return outs

    # 模型训练
    def train(self, ins, true, epochs, step=100):
        '''
        :param ins: 网络输入
        :param true: 网络真实输出
        :param epochs: 迭代次数
        :param step: 输出步长
        :return: None
        '''
        for epoch in range(epochs):
            net_out = self(ins)
            self.backward(true, net_out)
            if epoch % step == 0:
                print('epoch', epoch, 'loss', self.mse(true, net_out), 'pred', net_out)
        print('epoch', epoch + 1, 'loss', self.mse(true, net_out), 'pred', net_out)

    def save_param(self, netname, fname='cherishbp.pickle'):
        '''
        :param netname: bp网络的名称
        :param fname: 保存参数文件名
        :return: None
        '''
        with open(fname, "wb") as file:
            pickle.dump(netname, file)

    # 反向传播
    def backward(self, t, p):
        # 对误差求导
        grad = self.mse.der(t, p)
        #反向更新，使用reversed函数进行层索引
        for layer in reversed(self.layers):
            grad = layer.update(grad, self.lr)

    def predict(self, ins):
        return self(ins)


# network = CherishBP()
#
# x = np.array([[10]])
# true = np.array([[0.1]])
# # 训练 启动！！！
# network.train(x, true, 600, 100)
#
# print(network.predict(x))
#
# network.save_param(network) #保存模型为pickle文件
# #加载模型
# with open("cherishbp.pickle", "rb") as file:
#     network = pickle.load(file)
# print(network.predict(x))#使用模型进行预测


