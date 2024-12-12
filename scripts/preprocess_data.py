# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for parsing the 3D-FRONT data scenes into numpy files in order
to be able to avoid I/O overhead when training our model.
"""
import argparse
import logging
import json
import os
import shutil
import sys
import pickle

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from utils import DirLock, ensure_parent_directory_exists, \
    floor_plan_renderable, floor_plan_from_scene, \
    get_textured_objects_in_scene, scene_from_args, render
import ast


sys.path.append("..")
from scene_synthesis.datasets import filter_function
from scene_synthesis.datasets.threed_front import ThreedFront
from scene_synthesis.datasets.threed_front_dataset import \
    dataset_encoding_factory
from scene_synthesis.datasets.parse_walls import get_walls
from utils import distance_point_to_line
from sklearn.cluster import DBSCAN
from collections import Counter
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def main(argv):
    parser = argparse.ArgumentParser(
        description="Prepare the 3D-FRONT scenes to train our model"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_3d_front_dataset_directory",
        help="Path to the 3D-FRONT dataset"
    )
    parser.add_argument(
        "path_to_3d_future_dataset_directory",
        help="Path to the 3D-FUTURE dataset"
    )
    parser.add_argument(
        "path_to_model_info",
        help="Path to the 3D-FUTURE model_info.json file"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--path_to_invalid_scene_ids",
        default="../config/invalid_threed_front_rooms.txt",
        help="Path to invalid scenes"
    )
    parser.add_argument(
        "--path_to_invalid_bbox_jids",
        default="../config/black_list.txt",
        help="Path to objects that ae blacklisted"
    )
    parser.add_argument(
        "--annotation_file",
        default="../config/bedroom_threed_front_splits.csv",
        help="Path to the train/test splits file"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="The size of the room along a side (default:3.1)"
    )
    parser.add_argument(
        "--dataset_filtering",
        default="threed_front_bedroom",
        choices=[
            "threed_front_bedroom",
            "threed_front_livingroom",
            "threed_front_diningroom",
            "threed_front_library"
        ],
        help="The type of dataset filtering to be used"
    )
    parser.add_argument(
        "--without_lamps",
        action="store_true",
        help="If set ignore lamps when rendering the room"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,-1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window"
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # # 删除原文件夹的所有物体(非常危险的代码！)  #不能删，因为现在文件夹下有聚类信息
    # if os.path.exists(args.output_directory):
    #     shutil.rmtree(args.output_directory)
    # # Check if output directory exists and if it doesn't create it
    # if not os.path.exists(args.output_directory):
    #     os.makedirs(args.output_directory)

    # Create the scene and the behaviour list for simple-3dviz
    scene = scene_from_args(args)

    with open(args.path_to_invalid_scene_ids, "r") as f:
        invalid_scene_ids = set(l.strip() for l in f)  # 事先准备好invalid scenes的txt，这里将txt中的所有id存好

    with open(args.path_to_invalid_bbox_jids, "r") as f:
        invalid_bbox_jids = set(l.strip() for l in f)

    with open(args.config_file, "r") as f:
        data_config = yaml.load(f, Loader=Loader)["data"]

    config = {
        "filter_fn":                 args.dataset_filtering,
        "min_n_boxes":               -1,
        "max_n_boxes":               -1,
        "path_to_invalid_scene_ids": args.path_to_invalid_scene_ids,
        "path_to_invalid_bbox_jids": args.path_to_invalid_bbox_jids,
        "annotation_file":           args.annotation_file,  # _splits.csv，给rooms打上train或test的标签
        "data":                      data_config
    }
    # Initially, we only consider the train split to compute the dataset
    # statistics, e.g the translations, sizes and angles bounds
    # # 统计墙的最大面数
    # data_all = ThreedFront.from_dataset_directory(
    #     dataset_directory=args.path_to_3d_front_dataset_directory,
    #     path_to_model_info=args.path_to_model_info,
    #     path_to_models=args.path_to_3d_future_dataset_directory,
    #     filter_fn=filter_function(config, ["train","val","test"], args.without_lamps)
    # )
    # max_walls = 0
    # for d in data_all:
    #     num = len(d.walls["length"])
    #     if num > max_walls:
    #         max_walls = num
    # print(max_walls)

    # # 统计数据集信息
    # dataset_train = ThreedFront.from_dataset_directory(
    #     dataset_directory=args.path_to_3d_front_dataset_directory,
    #     path_to_model_info=args.path_to_model_info,
    #     path_to_models=args.path_to_3d_future_dataset_directory,
    #     filter_fn=filter_function(config, ["train"], args.without_lamps)
    # )
    # dataset_val = ThreedFront.from_dataset_directory(
    #     dataset_directory=args.path_to_3d_front_dataset_directory,
    #     path_to_model_info=args.path_to_model_info,
    #     path_to_models=args.path_to_3d_future_dataset_directory,
    #     filter_fn=filter_function(config, ["val"], args.without_lamps)
    # )
    # dataset_test = ThreedFront.from_dataset_directory(
    #     dataset_directory=args.path_to_3d_front_dataset_directory,
    #     path_to_model_info=args.path_to_model_info,
    #     path_to_models=args.path_to_3d_future_dataset_directory,
    #     filter_fn=filter_function(config, ["test"], args.without_lamps)
    # )
    # print("check  data")

    # 使用训练数据存储dataset_stats
    dataset = ThreedFront.from_dataset_directory(
        dataset_directory=args.path_to_3d_front_dataset_directory,
        path_to_model_info=args.path_to_model_info,
        path_to_models=args.path_to_3d_future_dataset_directory,
        filter_fn=filter_function(config, ["train", "val"], args.without_lamps)
    )

    print("Loading dataset with {} rooms".format(len(dataset)))

    # Compute the bounds for the translations, sizes and angles in the dataset.
    # This will then be used to properly align rooms.
    # bounds的意思就是最大值和最小值，边界
    tr_bounds = dataset.bounds["translations"]
    si_bounds = dataset.bounds["sizes"]
    an_bounds = dataset.bounds["angles"]

    dataset_stats = {
        "bounds_translations": tr_bounds[0].tolist() + tr_bounds[1].tolist(),
        "bounds_sizes": si_bounds[0].tolist() + si_bounds[1].tolist(),
        "bounds_angles": an_bounds[0].tolist() + an_bounds[1].tolist(),
        "class_labels": dataset.class_labels,
        "object_types": dataset.object_types,
        "class_frequencies": dataset.class_frequencies,
        "class_order": dataset.class_order,
        "count_furniture": dataset.count_furniture
    }
    # 将训练数据的信息保存到data/bedroom/dataset_stats.txt
    path_to_json = os.path.join(args.output_directory, "dataset_stats.txt")
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    with open(path_to_json, "w") as f:
        json.dump(dataset_stats, f)
        print(
            "Saving training statistics for dataset with bounds: {} to {}".format(
                dataset.bounds, path_to_json
            )
        )
    # 比之前多了'test'的数据,所有数据
    dataset = ThreedFront.from_dataset_directory(
        dataset_directory=args.path_to_3d_front_dataset_directory,
        path_to_model_info=args.path_to_model_info,
        path_to_models=args.path_to_3d_future_dataset_directory,
        filter_fn=filter_function(
            config, ["train", "val", "test"], args.without_lamps
        )
    )
    print(dataset.bounds)
    print("Loading dataset with {} rooms".format(len(dataset)))

    encoded_dataset = dataset_encoding_factory(
        "basic", dataset, augmentations=None, box_ordering=None
    )  # 将ThreedFront的数据转换成DatasetCollection

    for (i, es), ss in tqdm(zip(enumerate(encoded_dataset), dataset)):
        # es为每个room数据的家具信息，ss为每个room数据的信息
        # Create a separate folder for each room
        room_directory = os.path.join(args.output_directory, ss.uid)
        if not os.path.exists(room_directory):
            os.makedirs(room_directory)

        uids = [bi.model_uid for bi in ss.bboxes]  # uid: 44317/model
        jids = [bi.model_jid for bi in ss.bboxes]  # jid: 4e0171bf-e343-4963-921d-86022ad4a063

        floor_plan_vertices, floor_plan_faces, flg = ss.floor_plan
         # walls = get_walls(floor_plan_vertices, floor_plan_faces, np.copy(ss.floor_plan_centroid))
        walls = ss.walls
        # 就在这里聚类吧，将聚类信息存入walls.npz
        box_num = es["class_labels"].shape[0]
        walls_num = len(walls["length"])
        walls_points = np.stack(walls["points"])

        # # debug
        # if ss.scene_id == 'LivingDiningRoom-5442':
        #     print(walls["points"])

        trans = es["translations"]
        class_labels = es["class_labels"]

        # 第一步，计算距离分配墙
        walls_order = []  # 与原物体序列对应的最近墙序列
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
        walls_order = np.array(walls_order)  # 此时物体序列为默认顺序，对应该walls_order
        # 第二步，对场景聚类，看是否存在组合
        # # 读取聚类信息
        if "cluster" in config["data"]:
            cluster_str = config["data"]["cluster"]
            lines = [line.strip() for line in cluster_str.split("\n")]
            cluster_stat = [ast.literal_eval(line)[0] for line in lines]
            cluster_eps = float(config["data"]["eps"])
            cluster_samples = int(config["data"]["min_samples"])

        dbscan = DBSCAN(eps=cluster_eps, min_samples=cluster_samples)
        classes = np.array(dataset_stats["class_labels"])[class_labels.argmax(-1)]
        labels = dbscan.fit_predict(trans)
        # 该场景中的聚类
        cluster_idx = []
        for i in range(max(labels) + 1):
            cluster = Counter(list(classes[labels == i]))
            sort = sorted(cluster.items(), key=lambda x: x[0])  # 对键排序
            cluster = Counter(dict(sort))
            # 到统计数据的聚类中去查找，是否存在该组合
            for clus_data in cluster_stat:
                clus_data = Counter({clus[0]: clus[1] for clus in clus_data})
                if cluster == clus_data:  # 一致,保存该类别的索引
                    a = np.where(labels == i)[0]  # 获取同一类在场景中的索引，具体是那一类不需要知道
                    cluster_idx.append(a)
        if len(cluster_idx) > 0:  # 如果有统计中的类别，将涉及的物体分类到同一面墙
            for clus in cluster_idx:
                # 计算组别的中心坐标
                center=np.zeros(3)
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

        ss.walls["walls_order"] = walls_order
        # 保存墙的信息
        np.savez_compressed(
            os.path.join(room_directory, "walls"),
            walls=ss.walls
        )

        # 保存room_mask.png
        room_mask = render(
            scene,
            [floor_plan_renderable(ss)],  # 将floorplan的中心移到坐标原点，返回simple-3dviz renderable mesh
            (1.0, 1.0, 1.0),
            "flat",
            os.path.join(room_directory, "room_mask.png")
        )[:, :, 0:1]  # 取room_mask的第一维，墙内值为255，墙外值为0

        # 保存boxes.npz
        np.savez_compressed(
            os.path.join(room_directory, "boxes"),
            uids=uids,
            jids=jids,
            scene_id=ss.scene_id,
            scene_uid=ss.uid,
            scene_type=ss.scene_type,
            json_path=ss.json_path,
            room_layout=room_mask,
            floor_plan_vertices=floor_plan_vertices,
            floor_plan_faces=floor_plan_faces,
            floor_plan_centroid=ss.floor_plan_centroid,
            class_labels=es["class_labels"],
            translations=es["translations"],
            sizes=es["sizes"],
            angles=es["angles"],
        )

        # # 保存rendered_scene_256.png
        # path_to_image = "{}/rendered_scene_{}.png".format(
        #     room_directory, args.window_size[0]
        # )
        # # 为了debug暂时注释
        # if os.path.exists(path_to_image):
        #     continue
        #
        # # Get a simple_3dviz Mesh of the floor plan to be rendered
        # floor_plan, _, _ = floor_plan_from_scene(  # 地板
        #     ss, args.path_to_floor_plan_textures, without_room_mask=True
        # )
        # renderables = get_textured_objects_in_scene(  # 家具
        #     ss, ignore_lamps=args.without_lamps
        # )
        # render(
        #     scene,
        #     renderables + floor_plan,  # 地板+家具
        #     color=None,
        #     mode="shading",
        #     frame_path=path_to_image
        # )

if __name__ == "__main__":
    main(sys.argv[1:])
