import numpy as np
from sklearn.linear_model import LinearRegression
import random
import math
from tqdm import tqdm
from tools.geometry import *


def linear_fit_and_predict(data):
    # 提取 f 和 x 的列表
    f = [point[0] for point in data]
    x = [point[1] for point in data]

    # 使用 numpy 的 polyfit 函数进行线性拟合，得到拟合参数
    a, b = np.polyfit(x, f, 1)

    # 定义一个内嵌函数，用于给定 f 值计算对应的 x
    def predict(f_value):
        # 根据线性方程 f = ax + b，求解 x = (f - b) / a
        if a != 0:
            return (f_value - b) / a
        else:
            return b
    return predict


def ransac(points, iterations=100, sigma=10.0):
    points = np.array(points)
    points_num = points.shape[0]
    bestScore = -1
    choose_points = []
    for k in range(iterations):

        i1, i2 = random.sample(range(points_num), 2)
        p1 = points[i1]
        p2 = points[i2]

        score = 0
        for i in range(points_num):
            p = points[i]
            dis = point_distance_line(p, [p1, p2])
            if math.fabs(dis) < sigma:
                score += 1
        if score > bestScore:
            choose_points = [p1, p2]
            bestScore = score
        # print(f'loop-{k}', bestScore, points_num)
    p1, p2 = choose_points
    in_points = []
    out_points = []
    for i in range(points_num):
        p = points[i]
        dis = point_distance_line(p, [p1, p2])
        if math.fabs(dis) < sigma:
            in_points.append(p)
        else:
            out_points.append(p)
    return choose_points, in_points, out_points


def calc_sigma(p, sigma, gradation_scope):
    y = p[1]
    h = gradation_scope[1] - gradation_scope[0]
    h1 = y - gradation_scope[0]
    sigma_range = sigma[1] - sigma[0]
    return h1 / h * sigma_range


def ransac_gradation(points, iterations=100, sigma=None, gradation_scope=None):
    points = np.array(points)
    points_num = points.shape[0]
    print(points_num)
    bestScore = -1
    choose_points = []
    for it in tqdm(range(iterations)):
        if (it + 1) * 2 > points_num:
            break
        i1, i2 = random.sample(range(points_num), 2)
        p1 = points[i1]
        p2 = points[i2]
        score = 0
        for i in range(points_num):
            p = points[i]
            dis = point_distance_line(p, [p1, p2])
            # print(p,calc_sigma(p,sigma,gradation_scope))
            if math.fabs(dis) < calc_sigma(p, sigma, gradation_scope):
                score += 1
        if score > bestScore:
            choose_points = [p1, p2]
            bestScore = score
        # print(f'loop-{it}', bestScore, points_num)
    p1, p2 = choose_points
    in_points = []
    out_points = []
    for i in range(points_num):
        p = points[i]
        dis = point_distance_line(p, [p1, p2])
        if math.fabs(dis) < calc_sigma(p, sigma, gradation_scope):
            in_points.append(p)
        else:
            out_points.append(p)
    return choose_points, in_points, out_points


def linear_regression(pts, method=1):
    pts = np.array(pts)
    SAMPLE_NUM = len(pts)
    X = pts[:, 0]
    Y = pts[:, 1]
    theta = np.polyfit(X, Y, deg=2)
    a_, b_ = 0, 0
    x0 = np.array(list(map(lambda x: x ** 1, X)))
    x1 = np.ones(len(X))
    # shape=(SAMPLE_NUM,3)
    A = np.stack((x0, x1), axis=1)
    b = np.array(Y).reshape((SAMPLE_NUM, 1))
    # print("方法列表如下:"
    #       "1.最小二乘法 least square method "
    #       "2.线性回归法 Linear regression")
    # method = int(input("method="))

    if method == 1:
        theta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        theta = theta.flatten()
        a_ = theta[0]
        b_ = theta[1]
        # print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))
        # Y_predict = list(map(lambda x: a_ * x + b_, X))
    elif method == 2:
        # 利用线性回归构建模型,拟合数据
        model = LinearRegression()
        X_normalized = A
        Y_noise_normalized = np.array(Y).reshape((SAMPLE_NUM, 1))  #
        model.fit(X_normalized, Y_noise_normalized)
        # 利用已经拟合到的模型进行预测
        Y_predict = model.predict(X_normalized)
        a_ = model.coef_.flatten()[0]
        b_ = model.intercept_[0]
        # print(model.coef_)
        # print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))
    return a_, b_


if __name__ == '__main__':
    # ransac()
    print('-------------')
