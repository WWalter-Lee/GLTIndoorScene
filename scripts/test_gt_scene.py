#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""
用生成场景的代码，根据实际场景数据的参数，来展示场景；
之后可以加上序列排序，并且往场景中一个个加物体来展示，以此检查序列顺序
"""
import argparse
import copy
import logging
import os
import sys
from PIL import Image

import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene, check_gt_bbx

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects
from scene_synthesis.datasets.parse_walls import get_walls

from simple_3dviz import Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render
from check_pred import check_rough_pred, check_fine_pred
from sklearn.cluster import DBSCAN
from collections import Counter
import pickle
from utils import distance_point_to_line


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--n_sequences",
        default=10,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,-1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    # Build the dataset of 3D models # 从pickle的家具读取
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split= ["train", "val", "test"]
        ),
        split=["train", "val", "test"]
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    # # 统计整个数据集聚类信息
    # # 第一步，对每个场景聚类，保存到pkl文件中
    # dbscan = DBSCAN(eps=1.6, min_samples=2)  # 最大(邻域)半径和最小样本数,不同数据集要用不同的参数，主要是改半径
    # cluster = []
    # for num in range(len(raw_dataset)):
    #     print(num+1,"/",len(raw_dataset))
    #     data = raw_dataset[num]
    #     trans = data.translations  # 场景内的点
    #     classes = np.array(dataset.class_labels)[data.class_labels.argmax(-1)]  # 每个点对应的物体标签
    #     labels = dbscan.fit_predict(trans)
    #     # 每个场景中的聚类结果
    #     for i in range(max(labels) + 1):
    #         cluster_i = Counter(list(classes[labels == i]))
    #         cluster.append(cluster_i)
    # # 保存聚类结果，免得每次运行都要重新聚类
    # with open('../data/livingroom/cluster_livingroom_1_2.pkl', 'wb') as file:
    #     pickle.dump(cluster, file)

    # # 第二步，读取pkl，对组合按键排序，合并相同的组合
    # with open('../data/livingroom/cluster_livingroom_1_2.pkl', 'rb') as file:
    #     # 使用readlines方法读取文件的每一行
    #     cluster = pickle.load(file)
    # cluster_union = Counter()
    # for c in cluster:
    #     # c为一种组合,先对c按键排序
    #     sort = sorted(c.items(), key=lambda x: x[0])
    #     c = Counter(dict(sort))
    #     # 转换为tuple才能将整个组合作为键
    #     c_tuple = tuple(c.items())
    #     cluster_union.update({c_tuple})
    #     # print(1)
    #
    # # 第三步，取10个最常见的组合，并输出到txt文件
    # common_cluster = cluster_union.most_common(10)
    # # print(*common_cluster,sep="\n")
    # with open('../data/livingroom/cluster_common_10.txt', 'w') as file:  # 将10个最常见的组合
    #     for item in common_cluster:
    #         file.write(str(item) + '\n')
    #
    # # 第四步，手动筛选合理组合(同一面墙生成)，手动创建txt文件并保存
    # print("cluster over.")

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position

    for _ in range(10000):
        given_scene_id = None
        if args.scene_id:
            for i, di in enumerate(raw_dataset):  # 遍历raw dataset，找到scene_id和给定scene_id相同的数据的索引i，赋给given_scene_id；若没有则为None
                if str(di.scene_id) == args.scene_id:
                    given_scene_id = i

        classes = np.array(dataset.class_labels)  # 23，比正常的21种家具类型多了start和end

        if given_scene_id == None:  # 没找到要求生成的scene_id，则随机生成scene_idx
            scene_idx = np.random.choice(len(dataset))
        else:
            scene_idx = given_scene_id
        # scene_idx = 1477
        current_scene = raw_dataset[scene_idx]
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures
        )
        print("Using the {}th floor plan:".format(scene_idx), "scene_idx:", scene_idx)

        # walls_scale = dataset[scene_idx][0]["walls"]
        walls = get_walls(np.copy(current_scene.floor_plan_vertices), np.copy(current_scene.floor_plan_faces),np.copy(current_scene.floor_plan_centroid))
        bbox_params_t = np.concatenate([  # 将四个类别的参数cat成一个ndarray
            current_scene.class_labels,  # 23
            current_scene.translations,  # 3
            current_scene.sizes,  # 3
            current_scene.angles  # 1
        ], axis=-1)[None, :, :]

        ordering, walls_order = permute_center(walls, bbox_params_t, raw_dataset)  # 以物体中心距离所有墙的最小垂直排序

        # 直接用存好的数据的ordering
        # 要用存好的walls.npz数据的话，就不能用gew_walls,其他的points啥的也要用walls.npz中的，要么就老实的聚类
        # n_box = bbox_params_t.shape[1]
        # ordering = np.arange(n_box)
        # # 直接冲到文件夹中去读walls.npz
        # walls_path = current_scene.image_path.split("render")[0]+"walls.npz"
        # W = np.load(walls_path, allow_pickle=True)
        # walls = W["walls"].item()
        # walls_order = walls["walls_order"]

        bbox_params_t = bbox_params_t[0][ordering][None, :, :]

        floor_points = np.concatenate(walls["points"])
        wall_points = np.stack(walls["points"])[walls_order]

        # 检查物体bbx
        # check_gt_bbx(bbox_params_t, floor_points)

        # 加入start symbol
        start_symbol = np.zeros([1, 1, bbox_params_t.shape[2]])
        bbox_params_t = np.concatenate((start_symbol, bbox_params_t), axis=1)

        # 用matplotlib绘出gt的物体位置和类别图
        trans = bbox_params_t[:, 1:, -7:-4]  # 不要start
        labels = classes[bbox_params_t[0, 1:, :-7].argmax(-1)]
        angles = bbox_params_t[:, 1:, -1]  # 不要start

        # # 验证wacth tv的人体工学los
        # # scene_idx = 3116稍微有点偏
        # # scene_idx = 1477很正
        # tv_idx = np.where(np.char.find(labels, "tv_stand") >= 0)[0]
        # bed_idx = np.where(np.char.find(labels, "bed") >= 0)[0]
        # trans_tv = np.delete(trans[0][tv_idx[0]], 1)  # 删除y轴
        # trans_bed = np.delete(trans[0][bed_idx[0]], 1)
        # v = trans_tv - trans_bed  # 床指向电视机的向量
        # v /= np.linalg.norm(v)
        # # 修改角度来检查los
        # # bbox_params_t[:, 1:, -1][0][bed_idx] -= 0.2
        # angle_bed = angles[0][bed_idx]  # 参考旋转轴为(0,1)
        # # 构建旋转矩阵
        # angle_bed = - angle_bed # 控制旋转方向，顺时针还是逆时针
        # c,s = np.cos(angle_bed), np.sin(angle_bed)
        # R = np.array([[c, -s], [s, c]]).squeeze(-1)
        # u = np.dot(R,np.array([0,1]))  # 将参考ref(z轴)旋转角度
        # los = 1-((1+np.dot(u,v))/2)
        # # 正常是0.003479
        # # bed_angle += 0.03 轻微往tv方向转，期望los变小。结果：0.0019，检验正确
        # # bed_angle -= 0.2 往反方向转，期望los变大。结果：0.025，检验正确
        # print(1)


        # 验证desk和chair的距离，为了计算人体工学损失
        # if "desk" in labels and "chair" in labels:
        #     desk_id = labels.tolist().index("desk")
        #     chair_id = labels.tolist().index("chair")
        #     a = torch.from_numpy(current_scene.translations[desk_id])
        #     b = torch.from_numpy(current_scene.translations[chair_id])
        #     dis = torch.dist(a,b)  # 配套的距离确实在0.8以内，验证完毕
        #     print(1)



        cluster_para = (float(config["data"]["eps"]), int(config["data"]["min_samples"]))
        # check_rough_pred(trans, labels, floor_points, wall_points, cluster_para)

        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )

        renderables += floor_plan  # 加上地板的模型

        check_fine_pred(args, scene, renderables, classes, bbox_params_t)

        # 检查角度
        angle_min = -np.pi
        angle_max = np.pi
        angles = bbox_params_t[0, 1:, -1]
        # 转换相对角度
        ref_angles = []
        wall_basis = walls["basis"][walls_order]
        for i in range(len(angles)):
            basis_x = wall_basis[i][0]  # 当前物体对应墙的x轴的基
            x = np.array([1, 0, 0])
            cos_angle = np.dot(basis_x, x)  # 这里顺序无关 #因为都是基向量，所有模都为1
            theta = np.arccos(cos_angle)  # 计算的是要旋转的真实角度(非缩放后的) #学原数据处理的*2
            # 计算旋转方向，叉乘
            ore = np.cross(basis_x, x)  # 顺序有关；假如为正，说明x到x_basis为顺时针，angle为正即可；假如为负，则相反
            if ore[1] < 0:
                theta = -theta
            # 计算ref用的是真实值计算
            ref = (angles[i] + theta - angle_min) % (2 * np.pi) + angle_min
            ref_angles.append(ref)
            # print(i, "th angle:%.2f ; ref_angle:%.2f" % angles[i] % ref)
            print(i, "th angle:%.2f" % angles[i], "ref_angle:%.2f" % ref)

        # 用simple_vis_3d将物体按顺序一个个渲染出来
        renderables_show = [renderables[-1]]
        for i in range(len(renderables)-1):
            renderables_show.append(renderables[i])
            show(
                renderables_show,
                behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                title="Generated Scene"
            )

def permute_center(walls, boxes, dataset):
    '''
    以物体中心，到所有墙中点的最小距离作为依据排序
    '''
    walls_points = np.stack(walls["points"])  # 所有墙的点
    trans = boxes[0, :, -7:-4]
    class_labels = boxes[0, :, 0:23]

    walls_num = walls_points.shape[0]
    box_num = boxes.shape[1]

    # 计算每个物体离每面墙的距离
    walls_order = []
    for i in range(box_num):  # 物体索引
        wall_idx = -1
        min_dis = 9999
        # 将该物体对每一面墙都求一个距离
        for num in range(walls_num):
            # 首先检查P是否在AB线段范围内
            P = trans[i]
            A = walls_points[num][0]
            B = walls_points[num][1]
            # AP·AB=|AP||AB|cos(a)，A,B与P形成的角度，其中一个为钝角则表示点p不在范围内
            if np.dot(P - A, B - A) < 0 or np.dot(P - B, A - B) < 0:
                continue
            dis = distance_point_to_line(trans[i], walls_points[num][0], walls_points[num][1])
            if dis < min_dis:
                wall_idx = num
                min_dis = dis
        walls_order.append(wall_idx)
    walls_order = np.array(walls_order)  # 此时物体序列为默认顺序，对应该wall_gt

    # 第二步：再对场景聚类，看是否存在组合
    # 有聚类信息后再考虑常见组对聚类的影响
    if hasattr(dataset, "cluster_eps"):
        cluster_eps = dataset.cluster_eps
        cluster_samples = dataset.cluster_samples
        dbscan = DBSCAN(eps=cluster_eps, min_samples=cluster_samples)
        # 注意聚类要用原尺寸
        classes = np.array(dataset.object_types)[class_labels.argmax(-1)]
        labels = dbscan.fit_predict(trans)
        # 该场景中的聚类
        cluster_idx = []
        for i in range(max(labels) + 1):
            cluster = Counter(list(classes[labels == i]))
            sort = sorted(cluster.items(), key=lambda x: x[0])  # 对键排序
            cluster = Counter(dict(sort))
            # 到统计数据的聚类中去查找，是否存在该组合
            for clus_data in dataset.cluster:
                clus_data = Counter({clus[0]: clus[1] for clus in clus_data})
                if cluster == clus_data:  # 一致,保存该类别的索引
                    a = np.where(labels == i)[0]  # 获取同一类在场景中的索引，具体是那一类不需要知道
                    cluster_idx.append(a)
        if len(cluster_idx) > 0:  # 如果有统计中的类别，将涉及的物体分类到同一面墙
            for clus in cluster_idx:
                # 计算组别的中心坐标
                center = np.zeros(3)
                for idx in clus:
                    center += trans[idx]
                center /= len(clus)  # 除以改组别的物体数量
                # 计算组别中心到墙的距离
                wall_idx = -1
                min_dis = 9999
                for num in range(walls_num):
                    # 同样检查P是否在AB线段范围内
                    P = center
                    A = walls_points[num][0]
                    B = walls_points[num][1]
                    if np.dot(P - A, B - A) < 0 or np.dot(P - B, A - B) < 0:
                        continue
                    dis = distance_point_to_line(center, walls_points[num][0], walls_points[num][1])
                    if dis < min_dis:
                        wall_idx = num
                        min_dis = dis
                # 循环结束后的wall_idx就是该组合对应的墙,修改原本的wall_gt
                walls_order[clus] = wall_idx
        print(len(cluster_idx), "个常见组")

    # 原序列
    ordering = np.arange(box_num)
    wall_gt_order = walls_order

    # # 打乱序列顺序
    # ordering = np.random.permutation(box_num)
    # wall_gt_order = walls_order[ordering]  # 与物体排序序列对应的墙序列顺序

    return ordering, wall_gt_order


if __name__ == "__main__":
    main(sys.argv[1:])
