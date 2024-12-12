#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Modifications Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

from collections import defaultdict
import numpy as np
import json
import os
import pickle

from .threed_front_scene import Asset, ModelInfo, Room, ThreedFutureModel, \
    ThreedFutureExtra
from scene_synthesis.datasets.parse_walls import get_walls

def parse_threed_front_scenes(
    dataset_directory, path_to_model_info, path_to_models,
    path_to_room_masks_dir=None,
        path_to_scene=None
):
    if not path_to_scene == None:  # 如果有.pkl
        with open(path_to_scene, 'rb') as f:
            scenes = pickle.load(f)  # 得到的是所有类型房间的结果
    else:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)  # ModelInfor,.../3D-FUTURE-model/model_info.json;原始模型数据
        model_info = mf.model_info  # dict，解析后的3d-future模型资产

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")
        ]  # dict, 存储到3D-FRONT每一个数据的路径
        scenes = []
        unique_room_ids = set()
        # Start parsing the dataset
        print("Loading dataset ", end="")
        for i, m in enumerate(path_to_scene_layouts):
            print(i+1,"/",len(path_to_scene_layouts))
            with open(m) as f:
                data = json.load(f)  # 打开3D-FRONT中的每一个json文件作为data
                # Parse the furniture of the scene
                furniture_in_scene = defaultdict()
                # 将标注"合理"的家具存入dict: furniture_in_scene
                for ff in data["furniture"]:
                    if "valid" in ff and ff["valid"]:
                        furniture_in_scene[ff["uid"]] = dict(
                            model_uid=ff["uid"],
                            model_jid=ff["jid"],
                            model_info=model_info[ff["jid"]]
                        )
                # Parse the extra meshes of the scene e.g walls, doors,
                # windows etc.
                meshes_in_scene = defaultdict()
                for mm in data["mesh"]:
                    meshes_in_scene[mm["uid"]] = dict(
                        mesh_uid=mm["uid"],
                        mesh_jid=mm["jid"],
                        mesh_xyz=np.asarray(mm["xyz"]).reshape(-1, 3),
                        mesh_faces=np.asarray(mm["faces"]).reshape(-1, 3),
                        mesh_type=mm["type"]
                    )

                # Parse the rooms of the scene
                scene = data["scene"]
                # Keep track of the parsed rooms
                rooms = []
                for rr in scene["room"]:  # rr表示一个房间
                    # Keep track of the furniture in the room
                    furniture_in_room = []
                    # Keep track of the extra meshes in the room
                    extra_meshes_in_room = []
                    # Flag to keep track of invalid scenes
                    is_valid_scene = True

                    for cc in rr["children"]:  # cc表示一个room中的一个家具或mesh(墙，门，窗)
                        if cc["ref"] in furniture_in_scene:  # 如果是合理的家具
                            tf = furniture_in_scene[cc["ref"]]  # ref 家具名称
                            # If scale is very small/big ignore this scene
                            if any(si < 1e-5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if any(si > 5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            furniture_in_room.append(ThreedFutureModel(  #
                               tf["model_uid"],
                               tf["model_jid"],
                               tf["model_info"],
                               cc["pos"],
                               cc["rot"],  # 角度向量
                               cc["scale"],
                               path_to_models
                            ))
                        elif cc["ref"] in meshes_in_scene:  # 如果是extra meshes of the scene e.g walls, doors,
                            mf = meshes_in_scene[cc["ref"]]
                            extra_meshes_in_room.append(ThreedFutureExtra(
                                mf["mesh_uid"],
                                mf["mesh_jid"],
                                mf["mesh_xyz"],
                                mf["mesh_faces"],
                                mf["mesh_type"],
                                cc["pos"],
                                cc["rot"],
                                cc["scale"]
                            ))
                        else:
                            continue
                    if len(furniture_in_room) > 1 and is_valid_scene:
                        # Check whether a room with the same instanceid has
                        # already been added to the list of rooms
                        if rr["instanceid"] not in unique_room_ids:
                            unique_room_ids.add(rr["instanceid"])
                            room = Room(
                                rr["instanceid"],                # scene_id
                                rr["type"].lower(),              # scene_type
                                furniture_in_room,               # bounding boxes
                                extra_meshes_in_room,            # extras e.g. walls
                                m.split("/")[-1].split(".")[0],  # json_path
                                path_to_room_masks_dir
                            )
                            # 加入墙的信息
                            floor_plan_vertices, floor_plan_faces, flg = room.floor_plan
                            if flg == 0:  # 无地板
                                room.walls = {}
                            else:
                                walls = get_walls(np.copy(floor_plan_vertices), np.copy(floor_plan_faces),np.copy(room.floor_plan_centroid))
                                room.walls = walls
                            # Add to the list
                            rooms.append(room)
                scenes.append(rooms)
            s = "{:5d} / {:5d}".format(i, len(path_to_scene_layouts))
            print(s, flush=True, end="\b"*len(s))
        scenes = sum(scenes, [])

        with open('../data/threed_front.pkl', 'wb') as f:
            pickle.dump(scenes, f)  # 将所有房间类型的数据保存为pickle文件，不应用过滤器

    return scenes


def parse_threed_future_models(
    dataset_directory, path_to_models, path_to_model_info
):
    if os.getenv("PATH_TO_3D_FUTURE_OBJECTS"):
        furnitures = pickle.load(
            open(os.getenv("PATH_TO_3D_FUTURE_OBJECTS"), "rb")
        )
    else:
        # Parse the model info
        mf = ModelInfo.from_file(path_to_model_info)
        model_info = mf.model_info

        path_to_scene_layouts = [
            os.path.join(dataset_directory, f)
            for f in sorted(os.listdir(dataset_directory))
            if f.endswith(".json")
        ]
        # List to keep track of all available furniture in the dataset
        furnitures = []
        unique_furniture_ids = set()

        # Start parsing the dataset
        print("Loading dataset ", end="")
        for i, m in enumerate(path_to_scene_layouts):
            with open(m) as f:
                data = json.load(f)
                # Parse the furniture of the scene
                furniture_in_scene = defaultdict()
                for ff in data["furniture"]:
                    if "valid" in ff and ff["valid"]:
                        furniture_in_scene[ff["uid"]] = dict(
                            model_uid=ff["uid"],
                            model_jid=ff["jid"],
                            model_info=model_info[ff["jid"]]
                        )
                # Parse the rooms of the scene
                scene = data["scene"]
                for rr in scene["room"]:
                    # Flag to keep track of invalid scenes
                    is_valid_scene = True
                    for cc in rr["children"]:
                        if cc["ref"] in furniture_in_scene:
                            tf = furniture_in_scene[cc["ref"]]
                            # If scale is very small/big ignore this scene
                            if any(si < 1e-5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if any(si > 5 for si in cc["scale"]):
                                is_valid_scene = False
                                break
                            if tf["model_uid"] not in unique_furniture_ids:
                                unique_furniture_ids.add(tf["model_uid"])
                                furnitures.append(ThreedFutureModel(
                                    tf["model_uid"],
                                    tf["model_jid"],
                                    tf["model_info"],
                                    cc["pos"],
                                    cc["rot"],
                                    cc["scale"],
                                    path_to_models
                                ))
                        else:
                            continue
            # s = "{:5d} / {:5d}".format(i, len(path_to_scene_layouts))
            # print(s, flush=True, end="\b"*len(s))
            print(i, "/", len(path_to_scene_layouts))
        print()
        with open("../data/threed_future_model.pkl", "wb") as f:
            pickle.dump(furnitures, f)

    return furnitures

def scale(x, minimum, maximum):
    X = x.astype(np.float32)
    X = np.clip(X, minimum, maximum)
    X = ((X - minimum) / (maximum - minimum))  # ->[0,1]
    X = 2 * X - 1  # ->[-1,1]
    return X
def descale(x, minimum, maximum):
    x = (x + 1) / 2  # ->
    x = x * (maximum - minimum) + minimum
    return x

def distance_point_to_line(P, A, B):
    '''
    计算点P到线段AB的距离
    '''
    AP = A-P  # 向量AP
    AB = B-A  # 向量AB

    # 叉乘计算面积
    S = np.linalg.norm(np.cross(AP,AB))  # |AP||AB|sin(a)，值为向量的模长

    AB_len = np.linalg.norm(AB)  # |AB|

    dis = S/AB_len  # |AP|sin(a)

    return dis

