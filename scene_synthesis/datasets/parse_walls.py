#
# Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

import trimesh
import numpy as np
def get_walls(vertices, faces, centroid):
    vertices -= centroid
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)  # y轴全是0
    plan = mesh.projected(normal=(0, 1, 0), max_regions=10000)  # 设大region是为了能满足所有数据(含奇特的)正常提取pkl

    if plan.vertices.shape[0] <= 2:
        print("墙的点小于等于2个")
        return {}
    if len(plan.entities) == 0:
        print("无墙的实体")
        return {}
    plan = plan.simplify()  # 投影之后，x和z的坐标会调换
    nodes_idx = plan.vertex_nodes  # [0,1],[1,2]...
    nodes_xz = np.dot(plan.vertices, np.array([[0, -1], [-1, 0]]))  # 将x和z的坐标调回来
    nodes = np.dot(nodes_xz, np.array([[1,0,0],[0,0,1]]))

    walls = {}
    walls["points"] = []
    walls["length"] = []
    for id in nodes_idx:
        # 一段墙起点和终点坐标
        points = np.array(nodes[id])
        st = points[0]
        ed = points[1]
        length = np.linalg.norm(ed - st)
        # 墙长度大于某个值时才保存该墙
        if length > 0.5:
            walls['points'].append(points)
            walls['length'].append(length)

    if len(walls["points"]) <= 2:
        return {}

    # 检查墙的第一段和最后一段斜率是否相等
    p1 = walls['points'][0]
    s1 = p1[0]
    e1 = p1[1]
    v1 = s1-e1

    p2 = walls['points'][-1]
    s2 = p2[0]
    e2 = p2[1]
    v2 = s2 - e2
    if np.linalg.norm(np.cross(v1, v2)) == 0 and np.all(e2 == s1):  # 0或180° 加上判定最后一段的终点为第一段起点
        walls['points'][0][0] = s2  # 将最后一段的起点赋给第一段的起点
        walls["length"][0] += walls["length"][-1]  # 最后一段的长度加到第一段
        # 删除最后一段
        walls["points"].pop()
        walls["length"].pop()

    # 加上墙的基和坐标原点
    points = np.stack(walls["points"])
    origin = np.sum(points, axis=1) / 2  # 计算每一面墙的坐标原点
    axis_x = points[:, 1, :] - points[:, 0, :]  # 终点-起点,以墙的方向为x轴
    x_length = np.linalg.norm(axis_x, axis=1)
    basis_x = (axis_x / x_length[:, None])  # x轴的基
    # 这里将y指向和原来一致的方向，方便计算旋转角度，会导致z轴指向墙外
    basis_y = np.array([[0, 1, 0]]).repeat(basis_x.shape[0], axis=0)
    # z轴通过x和y的叉乘求得
    basis_z = np.cross(basis_x, basis_y)
    basis = np.stack((basis_x, basis_y, basis_z), axis=1)
    # 将所有墙的坐标原点和轴写入样本
    walls["origin"] = origin
    walls["basis"] = basis  # 这里写入的轴的基还是一行表示一个轴的向量，实际使用时要转置成一列表示个一轴

    return walls

def get_walls_from_points(points):
    nodes = np.stack((points[:,0],np.zeros((len(points))),points[:,1]),axis=1)
    nodes_idx_0 = np.arange(len(nodes)-1)
    nodes_idx_1 = nodes_idx_0+1
    nodes_idx = np.stack((nodes_idx_0, nodes_idx_1),axis=1)  # [0,1],[1,2]...

    walls = {}
    walls["points"] = []
    walls["length"] = []
    for id in nodes_idx:
        # 一段墙起点和终点坐标
        points = np.array(nodes[id])
        st = points[0]
        ed = points[1]
        length = np.linalg.norm(ed - st)
        # 墙长度大于某个值时才保存该墙
        if length > 2:
            walls['points'].append(points)
            walls['length'].append(length)

    if len(walls["points"]) <= 2:
        return {}

    # 检查墙的第一段和最后一段斜率是否相等
    p1 = walls['points'][0]
    s1 = p1[0]
    e1 = p1[1]
    v1 = s1 - e1

    p2 = walls['points'][-1]
    s2 = p2[0]
    e2 = p2[1]
    v2 = s2 - e2
    a = e2 == s1
    if np.linalg.norm(np.cross(v1, v2)) == 0 and np.all(e2 == s1):  # 0或180° 并且加上判定最后一段的终点为第一段起点
        walls['points'][0][0] = s2  # 将最后一段的起点赋给第一段的起点
        walls["length"][0] += walls["length"][-1]  # 最后一段的长度加到第一段
        # 删除最后一段
        walls["points"].pop()
        walls["length"].pop()

    # 存入pkl的基和坐标原点有问题，训练时不能用
    # 加上墙的基和坐标原点
    points = np.stack(walls["points"])
    origin = np.sum(points, axis=1) / 2  # 计算每一面墙的坐标原点
    axis_x = points[:, 1, :] - points[:, 0, :]  # 终点-起点,以墙的方向为x轴
    x_length = np.linalg.norm(axis_x, axis=1)
    basis_x = (axis_x / x_length[:, None])  # x轴的基
    # 这里将y指向和原来一致的方向，方便计算旋转角度，会导致z轴指向墙外
    basis_y = np.array([[0, 1, 0]]).repeat(basis_x.shape[0], axis=0)
    # z轴就通过x和y的叉乘求得
    basis_z = np.cross(basis_x, basis_y)
    basis = np.stack((basis_x, basis_y, basis_z), axis=1)
    # 将所有墙的坐标原点和轴写入样本
    walls["origin"] = origin
    walls["basis"] = basis  # 这里写入的轴的基还是一行表示一个轴的向量，实际使用时要转置成一列表示个一轴

    return walls