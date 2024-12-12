# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys
from PIL import Image

import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene, scale_walls,export_scene

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.networks.ModelUse import ModelUse
from scene_synthesis.utils import get_textured_objects
from scene_synthesis.datasets.parse_walls import get_walls
from check_pred import check_rough_pred,check_fine_pred

from simple_3dviz import Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render
import ast
import shutil
import random
import string


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

    network_rough, network_fine, _, _, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network_rough.eval()
    network_fine.eval()
    modeluse = ModelUse(network_rough, network_fine)  # 整合两个网络的使用

    args.camera_position=[0.,8.,0.]
    args.up_vector = [0.,0.,-1]
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position

    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):  # 遍历raw dataset，找到scene_id和给定scene_id相同的数据的索引i，赋给given_scene_id；若没有则为None
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    classes = np.array(dataset.class_labels)  # 23，比正常的21种家具类型多了start和end
    for i in range(args.n_sequences):  # 10，预设的生成场景个数
        # scene_idx = given_scene_id or np.random.choice(len(dataset))
        if given_scene_id == None:            # 没找到要求生成的scene_id，则随机生成scene_idx
            scene_idx = np.random.choice(len(dataset))
        else:
            scene_idx = given_scene_id
        current_scene = raw_dataset[scene_idx]
        print("{} / {}: Using the {}th floor plan, scene {}".format(  # scene_idx表示当前scene的索引，scene_id表示当前scene的id名称
            i, args.n_sequences, scene_idx, current_scene.scene_id)
        )
        # Get a floor plan
        # TexturedMesh, Trimesh, Tensor
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures
        )
        # room_mask_check = room_mask.cpu().numpy()
        walls_ori = get_walls(np.copy(current_scene.floor_plan_vertices),np.copy(current_scene.floor_plan_faces),np.copy(current_scene.floor_plan_centroid))
        walls = scale_walls(walls_ori, dataset.bounds)  # 对墙的坐标和角度进行缩放
        # mask_check = room_mask.numpy()

        # 读取聚类信息
        cluster_str = config["data"]["cluster"]
        lines = [line.strip() for line in cluster_str.split("\n")]
        cluster = [ast.literal_eval(line)[0] for line in lines]

        rough_info, bbox_params = modeluse.generate_boxes(  # 因为预测的是分布，采样会导致不同结果
            walls=walls,
            room_mask=room_mask.to(device),
            bounds=dataset.bounds,
            floor_points = np.concatenate(walls_ori["points"]),
            classes=classes,
            cluster=cluster,
            device=device,
        )

        boxes = dataset.post_process(bbox_params) # 后处理，将缩放的参数还原回原尺寸
        bbox_params_t = torch.cat([  # 将四个类别的参数cat成一个ndarray
            boxes["class_labels"],  # 23
            boxes["translations"],  # 3
            boxes["sizes"],         # 3
            boxes["angles"]         # 1
        ], dim=-1).cpu().numpy()

        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )

        renderables += floor_plan  # 加上地板的模型
        trimesh_meshes += tr_floor
        out_path = args.output_directory

        random_char1 = random.choice(string.ascii_letters)
        room_type = args.weight_file.split("/")[2]
        #scene_path = os.path.join(out_path,room_type,args.scene_id)
        scene_path = os.path.join(out_path, room_type, str(scene_idx))
        if not os.path.exists(scene_path):
            os.makedirs(scene_path)

        behaviours_check = [
            LightToCamera(),
            SaveFrames(os.path.join(scene_path, str(i)+random_char1+".png"), 360)
        ]
        render(
            renderables=renderables,
            behaviours=behaviours_check,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            n_frames=args.n_frames,
            scene=scene
        )

        # 将每个实际(fine网络)生成的物体保存为图片，和rough输出到同一个文件夹中
        # check_fine_pred(args, scene, renderables, classes, bbox_params_t)
        if args.without_screen:
            # Do the rendering
            path_to_image = "{}/{}/{}_{}_{:03d}_{}".format(
                args.output_directory,
                args.scene_id,
                current_scene.scene_id,
                scene_idx,
                i,
                random_char1
            )
            behaviours = [
                LightToCamera(),
                SaveFrames(path_to_image+".png", 1)
            ]
            if args.with_rotating_camera:
                behaviours += [
                    CameraTrajectory(
                        Circle(
                            [0, args.camera_position[1], 0],
                            args.camera_position,
                            args.up_vector
                        ),
                        speed=1/360
                    ),
                    SaveGif(path_to_image+".gif", 1)
                ]
            render(
                renderables,
                behaviours=behaviours,
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                n_frames=args.n_frames,
                scene=scene
            )
        else:
            show(
                renderables,
                behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                title="Generated Scene"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
