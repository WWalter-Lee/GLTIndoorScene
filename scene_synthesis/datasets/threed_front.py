#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Modifications Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

import pickle
from collections import Counter, OrderedDict
from functools import lru_cache
import numpy as np
import json
import os

from PIL import Image

from .common import BaseDataset
from .threed_front_scene import Room
from .utils import parse_threed_front_scenes
import ast


class ThreedFront(BaseDataset):
    """Container for the scenes in the 3D-FRONT dataset.
        Arguments
        ---------
        scenes: list of Room objects for all scenes in 3D-FRONT dataset
    """
    def __init__(self, scenes, bounds=None):
        super().__init__(scenes)
        assert isinstance(self.scenes[0], Room)  # 准备将Room的数据类型转换成ThreedFront
        self._object_types = None
        self._room_types = None
        self._count_furniture = None
        self._bbox = None

        self._sizes = self._centroids = self._angles = None
        if bounds is not None:
            self._sizes = bounds["sizes"]
            self._centroids = bounds["translations"]
            self._angles = bounds["angles"]

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self.scenes), self.n_object_types
        )

    @property
    def bbox(self):
        """The bbox for the entire dataset is simply computed based on the
        bounding boxes of all scenes in the dataset.
        找所有Rooms中最大的和最小的坐标点作为数据集的bbx
        """
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:  # s为Room
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    def _centroid(self, box, offset):
        return box.centroid(offset)

    def _size(self, box):
        return box.size

    def _compute_bounds(self):
        _size_min = np.array([10000000]*3)
        _size_max = np.array([-10000000]*3)
        _centroid_min = np.array([10000000]*3)
        _centroid_max = np.array([-10000000]*3)
        _angle_min = np.array([10000000000])
        _angle_max = np.array([-10000000000])
        for s in self.scenes:  # s为room
            for f in s.bboxes:  # f为家具
                if np.any(f.size > 5):  # 该room的家具数量大于5
                    print(s.scene_id, f.size, f.model_uid, f.scale)
                centroid = self._centroid(f, -s.centroid)  # -s.centroid为offset
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(f), _size_min)
                _size_max = np.maximum(self._size(f), _size_max)
                _angle_min = np.minimum(f.z_angle, _angle_min)
                _angle_max = np.maximum(f.z_angle, _angle_max)
        self._sizes = (_size_min, _size_max)
        self._centroids = (_centroid_min, _centroid_max)
        self._angles = (_angle_min, _angle_max)

    @property
    def bounds(self):
        pos1 = self.centroids[0]
        pos2 = self.centroids[1]
        self.length = np.linalg.norm(pos1-pos2)
        return {
            "translations": self.centroids,
            "sizes": self.sizes,
            "angles": self.angles,
            "length": self.length
        }

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles

    @property
    def count_furniture(self):
        if self._count_furniture is None:
            counts = []
            for s in self.scenes:
                counts.append(s.furniture_in_room)
            counts = Counter(sum(counts, []))
            counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
            self._count_furniture = counts
        return self._count_furniture

    @property
    def class_order(self):
        return dict(zip(
            self.count_furniture.keys(),
            range(len(self.count_furniture))
        ))

    @property
    def class_frequencies(self):
        object_counts = self.count_furniture
        class_freq = {}
        n_objects_in_dataset = sum(
            [object_counts[k] for k, v in object_counts.items()]
        )
        for k, v in object_counts.items():
            class_freq[k] = object_counts[k] / n_objects_in_dataset
        return class_freq

    @property
    def object_types(self):
        if self._object_types is None:
            self._object_types = set()
            for s in self.scenes:
                self._object_types |= set(s.object_types)
            self._object_types = sorted(self._object_types)
        return self._object_types

    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def class_labels(self):
        return self.object_types + ["start", "end"]

    @classmethod
    def from_dataset_directory(cls, dataset_directory, path_to_model_info,
                               path_to_models, path_to_room_masks_dir=None,
                               path_to_bounds=None, filter_fn=lambda s: s):
        scenes = parse_threed_front_scenes(
            dataset_directory,
            path_to_model_info,
            path_to_models,
            path_to_room_masks_dir,
            # path_to_scene = "../data/threed_front.pkl"  # 第一次运行时注释；有了threed_front.pkl文件后解除注释
                                                        # threed_front.pkl是所有场景的rooms的数据，没有应用过滤器；物体.pkl是应用了过滤器的
        )
        bounds = None
        if path_to_bounds:
            bounds = np.load(path_to_bounds, allow_pickle=True)

        # 统计各个房间的bbx数量
        d={}
        for s in scenes:
            if not s.scene_type in d:  # 创建房间类型
                d[s.scene_type]={}
            if not len(s.bboxes) in d[s.scene_type]:
                d[s.scene_type][len(s.bboxes)]=1
            else:
                d[s.scene_type][len(s.bboxes)]+=1
        # a = [s for s in map(filter_fn, scenes) if s]  # 应用过滤器，将scenes中返回为true的场景存为list
        return cls([s for s in map(filter_fn, scenes) if s], bounds)  # 将Room数据转换为ThreedFront数据，损失了墙等信息


class CachedRoom(object):
    def __init__(
        self,
        scene_id,
        room_layout,
        floor_plan_vertices,
        floor_plan_faces,
        floor_plan_centroid,
        class_labels,
        translations,
        sizes,
        angles,
        image_path
    ):
        self.scene_id = scene_id
        self.room_layout = room_layout
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.image_path = image_path

    @property
    def floor_plan(self):
        flg=1
        return np.copy(self.floor_plan_vertices), \
            np.copy(self.floor_plan_faces), flg

    @property
    def room_mask(self):
        return self.room_layout[:, :, None]


class CachedThreedFront(ThreedFront):
    '''
        raw_dataset
    '''
    def __init__(self, base_dir, config, scene_ids):
        self._base_dir = base_dir
        self.config = config

        self._parse_train_stats("../"+config["train_stats"])

        # 读取聚类信息
        if "cluster" in config:
            cluster_str = config["cluster"]
            lines = [line.strip() for line in cluster_str.split("\n")]
            self.cluster = [ast.literal_eval(line)[0] for line in lines]
            self.cluster_eps = float(config["eps"])
            self.cluster_samples = int(config["min_samples"])

        # 存"../data/bedroom/3D-FRONT"中所需的文件夹路径
        self._tags = sorted([
            oi
            for oi in os.listdir(self._base_dir)
            # for oi in os.listdir(os.path.join(self._base_dir, "3D-FRONT"))
            if oi.split("_")[1] in scene_ids
        ])

        # 存所需数据的boxes.npz路径
        self._path_to_rooms = sorted([
            os.path.join(self._base_dir, pi, "boxes.npz")
            for pi in self._tags
        ])
        # 存所需数据的rendered_scene_256.png路径
        rendered_scene = "rendered_scene_256.png"

        path_to_rendered_scene = os.path.join(
            self._base_dir, self._tags[0], rendered_scene
        )
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_256_no_lamps.png"
        self._path_to_renders = sorted([
            os.path.join(self._base_dir, pi, rendered_scene)
            for pi in self._tags
        ])
        # 存所需数据的walls.npz路径
        self._path_to_walls = sorted([os.path.join(self._base_dir, pi, "walls.npz")
            for pi in self._tags])

    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        # room_layout = np.ones([512,512,1])*255
        # room_layout = np.full([512,512,1], fill_value=255)
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    @lru_cache(maxsize=32)
    def __getitem__(self, i):
        D = np.load(self._path_to_rooms[i])
        return CachedRoom(
            scene_id=D["scene_id"],
            room_layout=self._get_room_layout(D["room_layout"]),
            floor_plan_vertices=D["floor_plan_vertices"],
            floor_plan_faces=D["floor_plan_faces"],
            floor_plan_centroid=D["floor_plan_centroid"],
            class_labels=D["class_labels"],
            translations=D["translations"],
            sizes=D["sizes"],
            angles=D["angles"],
            image_path=self._path_to_renders[i]
        )

    def get_room_params(self, i):
        D = np.load(self._path_to_rooms[i])
        W = np.load(self._path_to_walls[i], allow_pickle=True)
        room = self._get_room_layout(D["room_layout"])
        room = np.transpose(room[:, :, None], (2, 0, 1))
        return {
            "room_layout": room,
            "class_labels": D["class_labels"],
            "translations": D["translations"],
            "sizes": D["sizes"],
            "angles": D["angles"],
            "walls": W["walls"].item()
        }

    def __len__(self):
        return len(self._path_to_rooms)

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self), self.n_object_types
        )

    def _parse_train_stats(self, train_stats):
        # 从dataset_stats.txt中解析获取整个训练数据集的统计数据
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (
            np.array(self._centroids[:3]),
            np.array(self._centroids[3:])
        )
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture
