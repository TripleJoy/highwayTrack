import math

from tools import *
import numpy as np
from numpy import matrix as mat
from camera_calibration.xyz_trans import *
from tools.geometry import *


def func(args, pts, vp1):
    f = args[0, 0]
    h = args[1, 0]
    params = [f, h, vp1]
    p1, p2, ori_c = pts
    pt_c, pt_w2_real = p1, p2

    pt1, pt2 = pt_w2_real, camera_to_world2_xy(pt_c, params, ori_c)
    d = dis_between_two_points(pt1, pt2)
    return d


def deriv(args, pt, n, vp1):  # 对函数求偏导
    args1 = args.copy()
    args2 = args.copy()
    args1[n, 0] -= 0.000001
    args2[n, 0] += 0.000001
    p1 = func(args1, pt, vp1)
    p2 = func(args2, pt, vp1)
    d = (p2 - p1) * 1.0 / 0.000002
    return d


def get_cal_params_by_lm(input_data, param_init):
    vp1, f, h = param_init
    x, y = input_data
    n = len(x)
    param_num = 2
    J = mat(np.zeros((n, param_num)))  # 雅克比矩阵
    fx = mat(np.zeros((n, 1)))  # f(x)  100*1  误差
    fx_tmp = mat(np.zeros((n, 1)))
    xk = mat([[f], [h]])  # 参数初始化
    lase_mse = 0
    step = 0
    u, v = 1, 2
    conve = 10000
    while conve:
        mse, mse_tmp = 0, 0
        step += 1
        for i in range(n):
            fx[i] = func(xk, x[i], vp1) - y[i]
            mse += fx[i, 0] ** 2

            for j in range(param_num):
                J[i, j] = deriv(xk, x[i], j, vp1)
        H = J.T * J + u * np.eye(param_num)
        dx = -H.I * J.T * fx
        xk_tmp = xk.copy()
        xk_ = xk.copy()
        xk_tmp += dx
        for i in range(n):
            fx_tmp[i] = func(xk_tmp, x[i], vp1) - y[i]
            mse_tmp += fx_tmp[i, 0] ** 2
        q = (mse - mse_tmp) / ((0.5 * dx.T * (u * dx - J.T * fx))[0, 0])
        if q > 0:
            s = 1.0 / 3.0
            v = 2
            mse = mse_tmp
            xk = xk_tmp
            temp = 1 - pow(2 * q - 1, 3)
            if s > temp:
                u = u * s
            else:
                u = u * temp
            if abs(mse - lase_mse) < 0.000001:
                break
            lase_mse = mse  # 记录上一个 mse 的位置
        else:
            u = u * v
            v = 2 * v
        conve -= 1
        f_,h_ = xk_[0, 0],xk_[1, 0],
        phi, theta = get_phi_and_theta(f_, vp1)
        # print("step = %d ,f= %6f, h =%6f, phi =%6f, theta =%6f, mse = %.8f"
        #       % (step,f_,h_,phi,theta,math.sqrt(abs(lase_mse)/n)))
        # print("%d ,%6f, %6f,%6f,%6f,%.8f"
        #       % (step, f_, h_, phi, theta, math.sqrt(abs(lase_mse) / n)))
    p = []
    for i in range(param_num):
        p.append(xk[i, 0])
    return p, step, lase_mse
