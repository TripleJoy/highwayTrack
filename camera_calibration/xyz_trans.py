import math
import numpy as np


def get_phi_and_theta(f, v1):
    u0, v0 = v1
    phi = math.atan(-v0 / f)
    theta = math.atan(-u0 * math.cos(phi) / f)
    return phi, theta


def camera_to_world1(pt_c, params):
    f, h, v1 = params
    phi, theta = get_phi_and_theta(f, v1)
    u, v = pt_c
    y0 = h * (f - v * math.tan(phi)) / (v + f * math.tan(phi))
    # x0 = math.cos(phi) * u * (y0 + h * math.tan(phi)) / f
    x0 = h * u / (math.cos(phi) * v + f * math.sin(phi))
    return [x0, y0]


def world1_to_world2(pt_w1, params, ori_c):
    t = camera_to_world1(ori_c, params)
    f, h, v1 = params
    phi, theta = get_phi_and_theta(f, v1)
    x_w1, y_w1 = pt_w1

    x_w1 -= t[0]
    y_w1 -= t[1]
    x_w2_ = x_w1 * math.cos(theta) + y_w1 * math.sin(theta)
    y_w2_ = - x_w1 * math.sin(theta) + y_w1 * math.cos(theta)
    # flag = 1
    # if y_w1 < 0:
    #     flag = -1
    # d = math.sqrt(x_w1 ** 2 + y_w1 ** 2)
    # if d == 0.0:
    #     x_w2 = x_w1
    #     y_w2 = y_w1
    # else:
    #     theta1 = math.acos(x_w1 / d)
    #     theta2 = flag * theta1 - theta
    #     x_w2 = d * math.cos(theta2)
    #     y_w2 = d * math.sin(theta2)

    return [x_w2_, y_w2_]


def camera_to_world2_xy(pt_c, params, ori_c, center_point=None):
    if center_point is not None:
        pt_c = [pt_c[0] - center_point[0], pt_c[1] - center_point[1]]
    pt_w1 = camera_to_world1(pt_c, params)
    pt_w2 = world1_to_world2(pt_w1, params, ori_c)
    return pt_w2


def world2_to_world1(pt_w2, params, ori_c):
    t = camera_to_world1(ori_c, params)
    f, h, v1 = params
    phi, theta = get_phi_and_theta(f, v1)
    x_w2, y_w2 = pt_w2
    # flag = 1
    # if y_w2 < 0:
    #     flag = -1
    # d = math.sqrt(x_w2 ** 2 + y_w2 ** 2)
    # if d == 0.0:
    #     x_w1 = x_w2
    #     y_w1 = y_w2
    # else:
    #     theta2 = math.acos(x_w2 / d)
    #     theta1 = flag * theta2 + theta
    #     x_w1 = d * math.cos(theta1)
    #     y_w1 = d * math.sin(theta1)
    # x_w1 += t[0]
    # y_w1 += t[1]

    x_w1_ = x_w2 * math.cos(theta) - y_w2 * math.sin(theta)
    y_w1_ = x_w2 * math.sin(theta) + y_w2 * math.cos(theta)
    x_w1_ += t[0]
    y_w1_ += t[1]
    return [x_w1_, y_w1_]


def world2_xyz_to_camera(pt_w2_xyz, params, ori_c, center_point):
    f, h, v1 = params
    x, y, z = pt_w2_xyz
    pt_w2_2d = [x, y]
    phi, theta = get_phi_and_theta(f, v1)
    x_w1, y_w1 = world2_to_world1(pt_w2_2d, params, ori_c)
    p_mat = np.mat([[f, 0, 0, 0],
                    [0, -f * math.sin(phi), -f * math.cos(phi), f * h * math.cos(phi)],
                    [0, math.cos(phi), -math.sin(phi), h * math.sin(phi)]])
    pt_w1_vec = np.array([[x_w1], [y_w1], [z], [1]])
    pt_c_vec = np.dot(p_mat, pt_w1_vec)
    x_c, y_c, z_c = pt_c_vec[0][0], pt_c_vec[1][0], pt_c_vec[2][0]
    x_c /= z_c
    y_c /= z_c
    return [float(x_c) + center_point[0], float(y_c) + center_point[1]]
