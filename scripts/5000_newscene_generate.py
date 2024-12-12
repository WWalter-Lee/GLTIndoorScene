#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""跑测试数据中的例子，获取结果图"""
import argparse
import logging
import os
import sys
from PIL import Image

import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene, floor_plan_renderable, scale_walls, render,export_scene
import utils

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.networks.ModelUse import ModelUse
from scene_synthesis.utils import get_textured_objects
from scene_synthesis.datasets.parse_walls import get_walls,get_walls_from_points

from simple_3dviz import Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render
from check_pred import check_rough_pred, check_fine_pred
import ast
import pickle
from simple_3dviz import Lines
from simple_3dviz.behaviours.movements import LightTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.trajectory import BackAndForth
import shutil
import random
import string


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )
    parser.add_argument(
        "test_data",
        help="evaluate/test data"
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
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
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
        default="0,1,0",
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
        "--test_id",
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

    # test data
    test_path = args.test_data
    roomtype = test_path.split("/")[-1]
    points_pkl_path = test_path+"/"+roomtype+"_points.pkl"
    with open(points_pkl_path, 'rb') as f:
        test_scenes_points = pickle.load(f)
    # 读faces序列
    faces_pkl_path = test_path + "/" + roomtype + "_faces.pkl"
    with open(faces_pkl_path, 'rb') as f:
        test_scenes_faces = pickle.load(f)

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )  # 从pickle的家具读取
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=["train", "val", "test"]
        ),
        split=["train", "val", "test"]
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network_rough, network_fine, _, _, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network_rough.eval()
    network_fine.eval()
    modeluse = ModelUse(network_rough, network_fine)  # 整合两个网络的使用

    # args.camera_position=[0.,8.,0.]  # preprocess的不是4吗，但是似乎确实是用8比较合理
    args.up_vector = [0.,0.,-1]

    classes = np.array(dataset.class_labels)  # 23，比正常的21种家具类型多了start和end

    # 缩放边界，都是正方形
    bounds = {"bedroom": ((-2.5, -2.5), (2.5, 2.5)),
              "diningroom": ((-5, -5), (5, 5)),
              "livingroom": ((-5, -5), (5, 5)),
              "study": ((-2, -2), (2, 2))
              }
    bounds = bounds[roomtype]
    bound = bounds[1][0]

    out_path = os.path.join(args.output_directory, roomtype)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    generated_scenes = []
    for i in range(5000):  # 5，每个场景生成5个结果，取最好的
        print(i)
        data_idx = np.random.choice(len(test_scenes_points))
        points = test_scenes_points[data_idx]

        target_size = tuple(map(int, config["data"]["room_layout_size"].split(",")))  # (64,64)
        mask_img = Image.open(os.path.join(test_path, roomtype+"_mask", str(data_idx)+".png")).resize(target_size)
        mask = np.array(mask_img)[:,:,0]
        mask = np.asarray(mask).astype(np.float32) / np.float32(255)  # (64,64)
        room_mask=torch.tensor(mask)[None,None,:,:]

        # 检查获取的墙信息是否和之前一致
        walls_ori = get_walls_from_points(points)
        if(len(walls_ori)==0):
            continue
        walls = scale_walls(walls_ori, dataset.bounds)

        # 读取聚类信息
        cluster_str = config["data"]["cluster"]
        lines = [line.strip() for line in cluster_str.split("\n")]
        cluster = [ast.literal_eval(line)[0] for line in lines]

        rough_info, bbox_params = modeluse.generate_boxes(  # 因为预测的是分布，采样会导致不同结果
            walls=walls,
            room_mask=room_mask.to(device),
            bounds=dataset.bounds,
            floor_points=np.concatenate(walls_ori["points"]),
            classes=classes,
            cluster=cluster,
            device=device,
        )

        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([  # 将四个类别的参数cat成一个ndarray
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy()

        generated_scenes.append(bbox_params_t)
    # 将生成的场景输出到pkl
    with open('generated_test_scenes_5000.pkl', 'wb') as file:
        pickle.dump(generated_scenes, file)

if __name__ == "__main__":
    main(sys.argv[1:])
