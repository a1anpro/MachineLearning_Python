# -*- coding: utf-8 -*-
# @Time : 2019/1/27 23:01
# @Author : Yanhui
# @File : LinearRegression_yanhui.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

def featureNormalization(x):
    '''
     标准化 / 归一化，目的是去除数据的量纲，让数据在同一尺度
    所用标准化方法为：Z - Score
    标准化: x_new = (x - mu) / sigma
    :param x: 需要标准化的数据
    :return: 标准化的数据，平均值mu，标准差sigma
    '''
    x_norm = np.array(x)
    feature_num = x.shape[1]
    mu = np.zeros((1, feature_num))  # x.shape[1]得到的是列，即特征的个数
    sigma = np.zeros((1, feature_num))

    mu = np.mean(x_norm, 0)  # 每一列(特征)的平均值(0=列，1=行)
    sigma = np.std(x_norm, 0)

    for i in range(x_norm.shape[1]):
        # 遍历列,标准化公式: x_new = (x - mu) / sigma
        x_norm[:, i] = (x_norm[:, i] - mu[i]) / sigma[i]

    return x_norm, mu, sigma

def loadFile(fileName, split, dataType):
    '''
    按split格式和dataType类型加载数据
    :param fileName: 文件名
    :param split: 分隔符
    :param dataType: 所需的数据类型
    :return: 加载的数据
    '''
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def computeCost(x, y, theta):
    '''
    返回损失值
    :param X:
    :param y:
    :param theta:
    :return:
    '''
    m = y.size
    # 解释为什么要在X之前加一个全为1的列：为了直接用点乘计算
    hofx = np.dot(x, theta)
    cost = np.power(hofx - y, 2).sum()  # 给每个元素平方
    cost /= 2 * m
    return cost


def gradientDescend(x, y, theta, alpha, num_iters):
    '''
    梯度下降算法：目标是通过“梯度下降”的方法得到拟合效果较好的theta
    :param X:训练样本中的x
    :param y:训练样本中的y
    :param theta:初始的theta，在损失函数图中用该方法调优
    :param alpha:学习率
    :param num_iters:迭代次数
    :return:
    '''
    m = y.size  # 训练样本数量
    n = theta.size  # theta的个数
    temp = np.zeros((n, num_iters))  # 暂存每次迭代计算的theta
    J_history = np.zeros(num_iters)  # 存储每次计算的损失值

    for i in range(0, num_iters):
        h = np.dot(x, theta)  # 用点乘把h(x)计算出来，得到(m,1)
        temp[:, i] = theta - ((alpha/m)*(np.dot(h-y, x)))
        theta = temp[:, i]
        J_history[i] = computeCost(x, y, theta)
        # print(J_history[i])
    return theta, J_history



def linearRegression(alpha=0.01, num_iters=400):
    '''
    线性回归函数，通过提供的数据
    :param alpha: 学习速率(learning rate)
    :param num_iters: 迭代次数
    :return:
    '''
    print("加载数据...")
    data = loadFile("data.txt", ",", np.float64)  # 读取数据
    x = data[:, 0:-1]  # 读取数据中的第一列到倒数第二列，作为hypothesis的x
    y = data[:, -1]  # 数据的最后一列是正确的y
    m = len(y)  # 训练集的样本个数

    x, mu, sigma = featureNormalization(x)  # 对数据进行归一化

    x = np.c_[np.ones((m, 1)), x]  # 在x前面加一列1,作为与常数theta相乘的元素

    print("梯度下降算法...")  # 得到目标函数的theta
    theta = np.zeros(x.shape[1])
    theta, J_history = gradientDescend(x, y, theta, alpha, num_iters)
    plotJ(J_history, num_iters)
    return mu, sigma, theta

# 画每次迭代代价的变化图
def plotJ(J_history, num_iters):
    x = np.arange(1, num_iters + 1)
    plt.plot(x, J_history)
    plt.xlabel("迭代次数")  # 注意指定字体，要不然出现乱码问题
    plt.ylabel("代价值")
    plt.title(u"代价随迭代次数的变化")
    plt.show()

def test_one():
    data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))
    X = data[:, 0]  # 取所有行的第一个数据
    y = data[:, 1]
    m = y.size  # 训练集的样本个数
    X = np.c_[np.ones(m), X]  # 第1列是theta0, np.c_按行扩展
    theta = np.zeros(2)  # 初始化拟合参数 theta

    # 一些参数的设置
    iterations = 1500
    alpha = 0.01

    # 计算和显示初试的损失值
    print("初始损失值:" + str(computeCost(X, y, theta)) + ' (正确值=32.07)')
    theta, J_history = gradientDescend(X, y, theta, alpha, iterations)
    print('用梯度下降得到的theta: ' + str(theta.reshape(2)))

    # plt.figure(0)
    # # line1 = plt.plot(X[:, 1], np.dot(X, theta), label="linear regression")
    # line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
    # plt.legend(handles=[line1])  # 图例
    # plt.show()

def test_LinearRegression():
    mu, sigma, theta = linearRegression(0.01, 400)

def plot_2d_line(x):
    '''
    在二维平面画线
    :param x:
    :return:
    '''
    ...
    # plt.plot(x[,: 0], x[:, 1])
    # plt.show()


if __name__ == "__main__":
    test_LinearRegression()