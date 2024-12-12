#
# Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

import numpy as np
import torch


def global2ref(basis, origin, boxes):
    '''
    global坐标系表示的是世界坐标系,以(0,0,0)为原点，三根原来的轴为坐标轴
    :param basis: (3,3),局部坐标系的基，每一列表示一根轴的基向量
    :param origin: (3,),局部坐标原点
    :param boxes: (num,3),要转换的所有点的绝对坐标,每一行表示一个物体
    :return: ref_boxes: (num,3),转换后所有点的局部坐标，每一行表示一个物体
    '''
    box_num = len(boxes)
    # 用齐次坐标的方法
    # 平移变换矩阵
    T_view = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 平移矩阵的前三列: (4,3)
    T_l = np.concatenate((-origin, [1]))[:, None]  # 平移矩阵的最后一列: (4,1)
    T_view = np.concatenate((T_view, T_l), axis=1)
    # 旋转变换矩阵
    R_view = basis.T  # basis每一列表示一根轴的向量，相当于R_view_^-1
    R_view = np.concatenate((R_view, np.array([[0, 0, 0]])))  # (4,3)
    R_view = np.concatenate((R_view, np.array([[0], [0], [0], [1]])), axis=1)  # (4,4)
    # 变换矩阵
    M = np.dot(R_view, T_view)
    # 将坐标转换为齐次坐标形式
    boxes_m = np.concatenate((boxes, np.ones((box_num, 1))), axis=1).T
    b = np.dot(M, boxes_m)
    ref_translations = (b.T[:, :-1])

    return ref_translations

def ref2global(basis, origin, ref_boxes):
    '''
    与global2ref的唯一区别在于最后的变换矩阵，本来是从绝对坐标到相对坐标的变换矩阵
    现在只要求个逆，就是相对坐标到绝对坐标的变换矩阵，输入是缩放后的数值
    :param basis: ndarray:(3,3)
    :param origin: ndarray:(3)
    :param ref_boxes: ndarray:(1,3)
    :return: global_translations: ndarray:(1,3)
    '''
    box_num = len(ref_boxes)
    # 用齐次坐标的方法
    # 平移变换矩阵
    T_view = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 平移矩阵的前三列: (4,3)
    T_l = np.concatenate((-origin, [1]))[:, None]  # 平移矩阵的最后一列: (4,1)
    T_view = np.concatenate((T_view, T_l), axis=1)
    # 旋转变换矩阵
    R_view = basis.T  # basis每一列表示一根轴的向量，相当于R_view_^-1
    R_view = np.concatenate((R_view, np.array([[0, 0, 0]])))  # (4,3)
    R_view = np.concatenate((R_view, np.array([[0], [0], [0], [1]])), axis=1)  # (4,4)
    # 变换矩阵
    M = np.dot(R_view, T_view)  # 顺序有关
    M = np.linalg.inv(M)
    # 将坐标转换为齐次坐标形式，每一列表示一个点
    boxes_m = np.concatenate((ref_boxes, np.ones((box_num, 1))), axis=1).T

    b = np.dot(M, boxes_m)
    global_translations = (b.T[:, :-1])

    return global_translations

def ref2global_tensor(basis, origin, ref_boxes):
    '''
    将ref2global的参数改成tensor的
    :param basis: ndarray:(3,3)
    :param origin: ndarray:(3)
    :param ref_boxes: ndarray:(1,3)
    :return: global_translations: ndarray:(1,3)
    '''
    device = basis.device

    box_num = len(ref_boxes)
    # 用齐次坐标的方法
    # 平移变换矩阵
    T_view = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],device=device)
    T_l = torch.concatenate((-origin, torch.tensor([1],device=device)))[:,None]
    T_view = torch.concatenate((T_view, T_l), axis=1)

    # 旋转变换矩阵
    R_view = basis.T  # basis每一列表示一根轴的向量，相当于R_view_^-1
    R_view = torch.concatenate((R_view, torch.tensor([[0, 0, 0]],device=device)))  # (4,3)
    R_view = torch.concatenate((R_view, torch.tensor([[0], [0], [0], [1]],device=device)), axis=1)  # (4,4)
    # 变换矩阵
    M = torch.mm(R_view, T_view)  # 顺序有关
    M = torch.inverse(M)
    # 将坐标转换为齐次坐标形式，每一列表示一个点
    boxes_m = torch.concatenate((ref_boxes, torch.ones((box_num, 1),device=device)), axis=1).T

    b = torch.mm(M, boxes_m)
    global_translations = (b.T[:, :-1])

    return global_translations