from numpy import matrix as mat
from tools.geometry import *


def func(args, line):
    vp1 = [args[0, 0],args[1, 0]]
    d = point_distance_line(vp1,line)
    return d


def deriv(args, line, n):  # 对函数求偏导
    args1 = args.copy()
    args2 = args.copy()
    args1[n, 0] -= 0.000001
    args2[n, 0] += 0.000001
    p1 = func(args1, line)
    p2 = func(args2, line)
    d = (p2 - p1) * 1.0 / 0.000002
    return d


def get_vp1_by_lm(input_data, vp1_init):
    x, y = input_data
    n = len(x)
    param_num = 2
    J = mat(np.zeros((n, param_num)))  # 雅克比矩阵
    fx = mat(np.zeros((n, 1)))  # f(x)  100*1  误差
    fx_tmp = mat(np.zeros((n, 1)))
    xk = mat(vp1_init)  # 参数初始化
    lase_mse = 0
    step = 0
    u, v = 1, 2
    conve = 10000
    while conve:
        mse, mse_tmp = 0, 0
        step += 1
        for i in range(n):
            fx[i] = func(xk, x[i]) - y[i]
            mse += fx[i, 0] ** 2

            for j in range(param_num):
                J[i, j] = deriv(xk, x[i], j)
        H = J.T * J + u * np.eye(param_num)
        dx = -H.I * J.T * fx
        xk_tmp = xk.copy()
        xk_tmp += dx
        for i in range(n):
            fx_tmp[i] = func(xk_tmp, x[i]) - y[i]
            mse_tmp += fx_tmp[i, 0] ** 2
        q = (mse - mse_tmp) / ((0.5 * dx.T * (u * dx - J.T * fx))[0, 0])
        # print(q,mse,mse_tmp)
        # print("step = %d,mse = %.8f" % (step, abs(mse)))
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

    # print("step = %d,mse = %.8f" % (step, abs(lase_mse)))
    p = []
    for i in range(param_num):
        p.append(xk[i, 0])
    return p, lase_mse
