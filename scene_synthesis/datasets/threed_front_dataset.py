#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Modifications Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

import numpy
import numpy as np

from functools import lru_cache
from scipy.ndimage import rotate
from .global_ref_transform import global2ref
import torch
from torch.utils.data import Dataset

from .utils import scale, descale, distance_point_to_line
from sklearn.cluster import DBSCAN
from collections import Counter

class DatasetDecoratorBase(Dataset):
    """A base class that helps us implement decorators for ThreeDFront-like
    datasets."""
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):  # 7(bbx其他维度)+23(bbx类别数量)
        return self.bbox_dims + self.n_classes

    @property
    def bbox_dims(self):
        raise NotImplementedError()

    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxOrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, box_ordering=None):
        super().__init__(dataset)
        self.box_ordering = box_ordering

    @lru_cache(maxsize=16)
    def _get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        if self.box_ordering is None:
            return scene.bboxes
        elif self.box_ordering == "class_frequencies":
            return scene.ordered_bboxes_with_class_frequencies(
                self.class_frequencies
            )
        else:
            raise NotImplementedError()


class DataEncoder(BoxOrderedDataset):
    """
    DataEncoder is a wrapper for all datasets we have
    """
    @property
    def property_type(self):
        raise NotImplementedError()


class RoomLayoutEncoder(DataEncoder):
    """Implement the encoding for the room layout as images."""
    @property
    def property_type(self):
        return "room_layout"

    def __getitem__(self, idx):
        img = self._dataset[idx].room_mask[:, :, 0:1]
        return np.transpose(img, (2, 0, 1))

    @property
    def bbox_dims(self):
        return 0


class ClassLabelsEncoder(DataEncoder):
    """Implement the encoding for the class labels."""
    @property
    def property_type(self):
        return "class_labels"

    def __getitem__(self, idx):
        # Make a local copy of the class labels
        classes = self.class_labels

        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        C = len(classes)  # number of classes
        class_labels = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            class_labels[i] = bs.one_hot_label(classes)
        return class_labels

    @property
    def bbox_dims(self):
        return 0


class TranslationEncoder(DataEncoder):
    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        # Get the scene
        scene = self._dataset[idx]
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        translations = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            translations[i] = bs.centroid(-scene.centroid)
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncoder(DataEncoder):
    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        L = len(boxes)  # sequence length
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes

    @property
    def bbox_dims(self):
        return 3


class AngleEncoder(DataEncoder):
    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        # Get the scene
        boxes = self._get_boxes(idx)
        # Get the rotation matrix for the current scene
        L = len(boxes)  # sequence length
        angles = np.zeros((L, 1), dtype=np.float32)
        for i, bs in enumerate(boxes):
            angles[i] = bs.z_angle
        return angles

    @property
    def bbox_dims(self):
        return 1


class DatasetCollection(DatasetDecoratorBase):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets


    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

    @staticmethod
    def collate_fn(samples):

        max_walls = samples[0][2]  # config中

        samples_temp = []  # 绝对坐标系
        ref_samples = []  # 相对坐标系
        for i in samples:
            samples_temp.append(i[0])
            ref_samples.append(i[1])
        samples = samples_temp

        # 绝对坐标数据处理
        key_set = {"class_labels", "translations", "sizes", "angles",
                   "class_labels_tr", "translations_tr", "sizes_tr", "angles_tr",
                   "walls_order", "walls_length",
                   "walls_order_tr", "walls_length_tr"}  # 4*(物体信息+物体tr) + 2*(墙信息+墙tr)
        # Compute the max length of the sequences in the batch
        max_length = max(sample["length"] for sample in samples)  # 找到该批次数据中的(现有)家具数量最大值
        padding_keys = {"class_labels", "translations", "sizes", "angles", "walls_order", "walls_length"}  # key set除了tr
        sample_params = {}

        #room_layout
        sample_params.update({
            "room_layout": np.stack([sample["room_layout"] for sample in samples], axis=0)
        })
        # 所有tr
        tr_set = key_set-padding_keys
        sample_params.update({
            k: np.stack([sample[k] for sample in samples], axis=0)
            for k in (tr_set)
        })
        # 输入
        padding_object_keys = {"class_labels", "translations", "sizes", "angles"}
        padding_walls_keys = padding_keys-padding_object_keys
        # 物体的4个输入
        sample_params.update({  # 将需要padding的部分，都用0填充到该样本批次的最大值
            k: np.stack([
                np.vstack([
                    sample[k],
                    np.zeros((max_length - len(sample[k]), sample[k].shape[1]))  # np.zeros(k属性需要填充多少个(len(sample[k])和sample[k].shape[0])等价,每个填充需要多少维度(例如类别是23))
                ]) for sample in samples  # 对样本批次中的每一个样本都进行填充操作
            ], axis=0)
            for k in padding_object_keys  # 需要padding的部分
        })
        # 墙的2个输入
        sample_params.update({
           k: np.stack([
                np.concatenate((sample[k], np.zeros(max_length - len(sample[k]))))
                for sample in samples
            ])
            for k in padding_walls_keys
        })
        # 修改walls_length的填充值为-1
        sample_params.update({
            "walls_length": np.stack([
                np.concatenate((sample["walls_length"], (-1) * np.ones(max_length - len(sample["walls_order"]))))
                for sample in samples
            ])
        })

        sample_params["lengths"] = np.array([
            sample["length"] for sample in samples
        ])  # 统计每个场景的家具数量length(不含end symbol)
        sample_params["walls_num"] = np.array([
            sample["walls_num"] for sample in samples
        ])  # 统计每个场景墙的数量(包含wall_0,也就是eos墙)

        # 收集场景中墙的数据(坐标+角度+长度+类别)，之后要用在相对坐标中的第一列
        k = "walls"
        # wall_points
        sample_params.update({  # 将需要padding的部分，都用0填充到该样本批次的最大值
            "wall_points_ref": np.stack([
                np.vstack([
                    sample[k]["points"],
                    np.zeros((max_walls - len(sample[k]["points"]), 2, 3))
                    # np.zeros(k属性需要填充多少个(len(sample[k])和sample[k].shape[0])等价,每个填充需要多少维度(例如类别是23))
                ]) for sample in samples  # 对样本批次中的每一个样本都进行填充操作
            ], axis=0)
        })
        # wall_length
        sample_params.update({
            "wall_length_ref": np.stack([
                np.concatenate([
                    sample[k]["length"],
                    -np.ones((max_walls - len(sample[k]["length"])))
                ]) for sample in samples
            ], axis=0)})

        # 加上eos墙，即索引为[0]的墙
        pad_points = np.zeros_like(sample_params["wall_points_ref"])[:, 0][:, None]
        pad_length = -np.ones_like(sample_params["wall_length_ref"])[:, 0][:, None]
        sample_params["wall_points_ref"] = np.concatenate((pad_points, sample_params["wall_points_ref"]), axis=1)
        sample_params["wall_length_ref"] = np.concatenate((pad_length, sample_params["wall_length_ref"]), axis=1)

        # 为了计算人体工学损失，新增返回墙的信息
        # basis
        sample_params.update({  # 将需要padding的部分，都用0填充到该样本批次的最大值
            "walls_basis": np.stack([
                np.vstack([
                    sample[k]["basis"],
                    np.zeros((max_walls - len(sample[k]["basis"]), 3, 3))
                ]) for sample in samples  # 对样本批次中的每一个样本都进行填充操作
            ], axis=0)
        })
        # origin
        sample_params.update({  # 将需要padding的部分，都用0填充到该样本批次的最大值
            "walls_origin": np.stack([
                np.vstack([
                    sample[k]["origin"],
                    np.zeros((max_walls - len(sample[k]["origin"]), 3))
                ]) for sample in samples  # 对样本批次中的每一个样本都进行填充操作
            ], axis=0)
        })
        # origin_scaled
        sample_params.update({  # 将需要padding的部分，都用0填充到该样本批次的最大值
            "walls_origin_scaled": np.stack([
                np.vstack([
                    sample[k]["origin_scaled"],
                    np.zeros((max_walls - len(sample[k]["origin_scaled"]), 3))
                ]) for sample in samples  # 对样本批次中的每一个样本都进行填充操作
            ], axis=0)
        })
        # 加上eos墙即索引为[0]的墙
        pad_basis = np.zeros_like(sample_params["walls_basis"])[:, 0][:, None]
        pad_origin = np.zeros_like(sample_params["walls_origin"])[:, 0][:, None]
        sample_params["walls_basis"] = np.concatenate((pad_basis, sample_params["walls_basis"]), axis=1)
        sample_params["walls_origin"] = np.concatenate((pad_origin, sample_params["walls_origin"]), axis=1)
        sample_params["walls_origin_scaled"] = np.concatenate((pad_origin, sample_params["walls_origin_scaled"]), axis=1)

        # 相对坐标信息的整理：相对坐标信息需要的就是四个输入和四个tr
        # 相对坐标系需要两个padding，一种是和上面一样的物体序列padding到max_length，第二种是墙的序列个数需要padding到max_walls
        ref_sample_params = {}
        max_walls += 1  # 绝对坐标时不加max_walls是因为在最后补充了eos墙，相对坐标这里之前已经加入了eos墙的信息，因此填充需要将wall_max+1
        for k in ref_samples[0].keys():
            if ref_samples[0][k].ndim == 2:  # tr数据
                ref_sample_params.update({
                    k: np.stack([
                        np.vstack([
                            sample[k],
                            np.zeros((max_walls - len(sample[k]), sample[k].shape[1]))
                        ]) for sample in ref_samples  # 对样本批次中的每一个样本都进行填充操作
                    ], axis=0)
                })
            elif ref_samples[0][k].ndim == 3:  # condition数据
                ref_sample_params.update({
                    k: np.stack([
                        np.concatenate([
                            np.vstack([
                                sample[k],
                                np.zeros((max_walls - len(sample[k]), sample[k].shape[1], sample[k].shape[2]))
                            ]),  # padding墙个数，补到相同的max_walls
                            np.zeros((max_walls, max_length-sample[k].shape[1], sample[k].shape[2]))
                        ], axis=1)  # padding物体个数
                        for sample in ref_samples  # 对样本批次中的每一个样本都进行填充操作
                    ], axis=0)
                })
        sample_params.update(ref_sample_params)
        # Make torch tensors from the numpy tensors
        torch_sample = {
            k: torch.from_numpy(sample_params[k]).float()
            for k in sample_params
        }

        # 将带有_tr的数据扩充第二维度，使其变成三维和输入相同
        for k in torch_sample.keys():
            if "_tr" in k and not k=="ref_translations":
                if "ref_" in k:
                    torch_sample.update({k: torch_sample[k][:, :, None]})  # 在[2]维插入一维
                else:
                    torch_sample.update({k: torch_sample[k][:, None]})
        # 返回之前，将walls_order相关的两个转换为one-hot
        torch_sample["walls_order_onehot"] = torch.nn.functional.one_hot(torch_sample["walls_order"].to(int), num_classes=max_walls).type(torch.FloatTensor)
        torch_sample["walls_order_tr_onehot"] = torch.nn.functional.one_hot(torch_sample["walls_order_tr"].to(int), num_classes=max_walls).type(torch.FloatTensor)

        return torch_sample


class CachedDatasetCollection(DatasetCollection):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

        self.walls_max = int(dataset.config["walls_max"])
        if hasattr(dataset, "cluster"):
            self.cluster = dataset.cluster  # 手动添加聚类信息
            self.cluster_eps = dataset.cluster_eps
            self.cluster_samples = dataset.cluster_samples

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class RotationAugmentation(DatasetDecoratorBase):
    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145):  # 弧度制 (0, 6.28)
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad

        self.walls_max = dataset.walls_max
        self.cluster = dataset.cluster  # 手动添加聚类信息
        self.cluster_eps = dataset.cluster_eps
        self.cluster_samples = dataset.cluster_samples


    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R  #

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)  # 一半的概率返回一个旋转角度
        else:
            return 0.0

    def __getitem__(self, idx):
        # Get the rotation matrix for the current scene
        rot_angle = self.rot_angle  # 一半的概率产生的随机角度,弧度制
        R = RotationAugmentation.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]

        for k, v in sample_params.items():
            if k == "translations":  # 坐标位置直接乘上旋转矩阵
                sample_params[k] = v.dot(R)
            elif k == "angles":
                angle_min, angle_max = self.bounds["angles"]  # -3.14和3.14
                sample_params[k] = \
                    (v + rot_angle - angle_min) % (2 * np.pi) + angle_min
            elif k == "walls":
                walls = v
                # 墙坐标旋转
                points = np.array(walls["points"])
                walls["points"] = points.dot(R)

            elif k == "room_layout":
                # Fix the ordering of the channels because it was previously changed
                img = np.transpose(v, (1, 2, 0))
                sample_params[k] = np.transpose(rotate(
                    img, rot_angle * 180 / np.pi, reshape=False
                ), (2, 0, 1))

        return sample_params


class Scale(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

        self.walls_max = dataset.walls_max
        if hasattr(dataset, "cluster"):
            self.cluster = dataset.cluster  # 手动添加聚类信息
            self.cluster_eps = dataset.cluster_eps
            self.cluster_samples = dataset.cluster_samples

    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))  # ->[0,1]
        X = 2 * X - 1  # ->[-1,1]
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds  # 整个数据集的统计
        sample_params = self._dataset[idx]

        # 获取墙的坐标原点(需要缩放)和坐标轴的基(不需要缩放)
        # 墙点连接的顺序是逆时针，以一面墙为例，假如以墙的方向为x轴，假如保持y轴向上不变，那么z轴就指向墙外；
        if type(sample_params["walls"]["points"]) == list:  # 是list说明是验证数据，不经过数据增强，传入数据格式为list
            sample_params["walls"]["points"] = np.stack(sample_params["walls"]["points"])
        points = sample_params["walls"]["points"]
        origin = np.sum(points, axis=1) / 2  # 计算每一面墙的坐标原点
        axis_x = points[:, 1, :] - points[:, 0, :]  # 终点-起点,以墙的方向为x轴
        x_length = np.linalg.norm(axis_x, axis=1)
        basis_x = (axis_x / x_length[:, None])  # x轴的基
        # 这里将y指向和原来一致的方向，方便计算旋转角度，z轴指向墙外
        basis_y = np.array([[0, 1, 0]]).repeat(basis_x.shape[0], axis=0)
        # z轴就通过x和y的叉乘求得
        basis_z = np.cross(basis_x, basis_y)
        basis = np.stack((basis_x, basis_y, basis_z), axis=1)
        # # 将所有墙的坐标原点和轴写入样本
        sample_params["walls"]["origin"] = origin
        sample_params["walls"]["basis"] = basis  # 这里写入的轴的基还是一行表示一个轴的向量，实际使用时要转置成列表示一个轴

        # 缩放
        for k, v in sample_params.items():
            if k in bounds:  # 对bbx的angle，size，和translation(位置)进行缩放
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]  # v:需要缩放的值，后面两项是缩放的min和max
                )
            if k == "walls":  # 只剩下墙长度需要scale
                len = np.copy(v["length"])
                v["length"] = Scale.scale(
                    len, 0, bounds["length"]
                )
                v["origin_scaled"] = Scale.scale(v["origin"], bounds["translations"][0],bounds["translations"][1])
        return sample_params

    def post_process(self, s):
        bounds = self.bounds  # 整个数据集的信息
        sample_params = {}
        for k, v in s.items():
            if k == "room_layout" or k == "class_labels":
                sample_params[k] = v
            elif k in bounds:
                sample_params[k] = Scale.descale(  # 数值类型的要进行缩放
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Jitter(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "room_layout" or k == "class_labels":
                sample_params[k] = v
            else:
                sample_params[k] = v + np.random.normal(0, 0.01)
        return sample_params


class Permutation(DatasetDecoratorBase):
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys  # ["class_labels", "translations", "sizes", "angles"]
        self._permutation_axis = permutation_axis  # 0 #表示序列顺序的轴

        self.walls_max = dataset.walls_max
        if hasattr(dataset, "cluster"):
            self.cluster = dataset.cluster  # 手动添加聚类信息
            self.cluster_eps = dataset.cluster_eps
            self.cluster_samples = dataset.cluster_samples

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["class_labels"].shape  # shapes[0]表示该场景物体的数量
        box_num = shapes[0]
        # 直接读取墙序列
        walls_order = sample_params["walls"]["walls_order"]
        walls_length = sample_params["walls"]["length"]

        # 打乱序列顺序
        ordering = np.random.permutation(box_num)
        wall_gt = walls_order[ordering]  # 与物体排序序列对应的墙序列顺序

        for k in self._permutation_keys:  # 四个输入
            sample_params[k] = sample_params[k][ordering]
        sample_params["walls_order"] = wall_gt
        sample_params["walls_length"] = walls_length[wall_gt]
        sample_params["walls"].pop("walls_order")

        return sample_params

class OrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, ordered_keys, box_ordering=None):
        super().__init__(dataset)
        self._ordered_keys = ordered_keys  # ["class_labels", "translations", "sizes", "angles"]
        self._box_ordering = box_ordering  # None

        self.walls_max = dataset.walls_max
        if hasattr(dataset, "cluster"):
            self.cluster = dataset.cluster  # 手动添加聚类信息
            self.cluster_eps = dataset.cluster_eps
            self.cluster_samples = dataset.cluster_samples

    def __getitem__(self, idx):
        if self._box_ordering is None:
            return self._dataset[idx]

        if self._box_ordering != "class_frequencies":
            raise NotImplementedError()

        sample = self._dataset[idx]
        order = self._get_class_frequency_order(sample)
        for k in self._ordered_keys:
            sample[k] = sample[k][order]
        return sample

    def _get_class_frequency_order(self, sample):
        t = sample["translations"]
        c = sample["class_labels"].argmax(-1)
        class_frequencies = self.class_frequencies
        class_labels = self.class_labels
        f = np.array([
            [class_frequencies[class_labels[ci]]]
            for ci in c
        ])

        return np.lexsort(np.hstack([t, f]).T)[::-1]


class Autoregressive(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.walls_max = dataset.walls_max
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        # 转换相对坐标
        walls = sample_params["walls"]  # 场景的所有墙信息
        walls_num = len(walls["points"])
        box_num = len(sample_params["translations"])

        # 由于用的boxes是缩放后的，因此用的墙的坐标原点也需要是缩放后的
        walls_origin = walls["origin_scaled"]  # 每面墙的中点(也是局部坐标原点)
        walls_basis = walls["basis"]  # 这里还是每一行表示一根轴的基
        ref_sample_params = {}  # 储存相对坐标系的信息

        for k, v in sample_params.items():
            if k == "class_labels" or k == "sizes":  # 如果是这两项只需要完全复制即可
                ref_sample_params["ref_"+k] = np.expand_dims(v, axis=0).repeat(walls_num,axis=0)
            elif k == "translations":
                # 一面面来
                ref_translations = []
                boxes = sample_params["translations"]  # (box_num,3)
                for i in range(walls_num):
                    wall_basis = walls_basis[i].T  # 转置成每一列表示一根轴的基
                    wall_origin = walls_origin[i]
                    b = global2ref(wall_basis, wall_origin, boxes)
                    ref_translations.append(b)
                ref_sample_params["ref_" + k] = np.stack(ref_translations)  # (walls_num,box_num,3),所有物体对每堵墙的相对坐标

            elif k =="angles":
                # 通过计算x轴旋转的角度，应用到每个物体即可
                # 同样还是一面面来
                angle_min = self.bounds["angles"][0]
                angle_max = self.bounds["angles"][1]
                ref_angles=[]
                for i in range(walls_num):
                    wall_basis_x = walls_basis[i][0]  # 每一行表示一根轴的基
                    x = np.array([1,0,0])
                    cos_angle = np.dot(wall_basis_x, x)  # 这里顺序无关
                    angle = np.arccos(cos_angle)  # 计算的是要旋转的真实角度(非缩放后的)
                    # 计算旋转方向，叉乘
                    ore = np.cross(wall_basis_x, x)  # 顺序有关；假如为正，说明x到x_basis为顺时针，angle为正即可；假如为负，则相反
                    if ore[1] < 0:
                        angle = -angle
                    # 先将v解缩放回真实值,但不改变原先的v
                    v_ori = descale(v, angle_min, angle_max)
                    # 计算ref用的是真实值计算
                    ref = (v_ori + angle - angle_min) % (2 * np.pi) + angle_min
                    # 计算后再将ref的真实值缩放到[-1,1]
                    ref = scale(ref, angle_min, angle_max)
                    ref_angles.append(ref)
                ref_sample_params["ref_" + k] = np.stack(ref_angles)  # (walls_num,box_num,angle,1)

        # 在相对坐标序列组中加上墙的end_symbol对应的序列
        sample_params["walls_order"] += 1  # 其他所有墙的索引+1(walls_length和walls_class不用变)
        for k, v in ref_sample_params.items():  # 四个相对参数(包括绝对的类别和尺寸)
            padding = numpy.zeros_like(v)[0][None, :, :]  # 取一维即可
            v = np.vstack((padding, v))  # 这里表示多一面eos墙(索引为0的墙)的序列
            ref_sample_params[k] = v

        # 加入物体的end symbol获取物体tgt序列，初始序列的操作不变，相对坐标序列也加上end symbol
        # _tr就是在原数据的基础上，加上一个end symbol
        sample_params_tgt = {}
        ref_sample_params_tgt = {}
        for k, v in sample_params.items():
            if k == "class_labels":
                class_labels = np.copy(v)
                L, C = class_labels.shape  # L:该场景的家具个数, C:类别总数量
                # Add the end label the end of each sequence
                end_label = np.eye(C)[-1]  # 创建一个one-hot的end label，最后一位是1
                sample_params_tgt[k+"_tr"] = np.vstack([
                    class_labels, end_label
                ])
                # 相对坐标序列，拓展end_label的维度
                end_label = np.expand_dims(end_label, axis=0).repeat(walls_num+1, axis=0).reshape(walls_num+1, 1, -1)  # (wall_num+1,1,26)
                ref_sample_params_tgt["ref_" + k + "_tr"] = np.concatenate((ref_sample_params["ref_" + k], end_label), axis = 1)
            elif k =="angles" or k =="translations" or k =="sizes":
                p = np.copy(v)
                # Set the attributes to for the end symbol
                _, C = p.shape  # C表示该变量(angles,translations,sizes)的维度
                sample_params_tgt[k+"_tr"] = np.vstack([p, np.zeros(C)])
                # 相对坐标序列
                ref_sample_params_tgt["ref_"+ k + "_tr"] = np.concatenate((ref_sample_params["ref_"+ k], np.zeros([walls_num+1,1,C])),axis=1)
            elif k == "walls_order" or k == "walls_length" or k == "walls_class":  # 绝对坐标下的墙信息
                sample_params_tgt[k + "_tr"] = np.append(v, 0)  # 墙的end symbol用0表示
                if k == "walls_length":
                    sample_params_tgt[k + "_tr"][-1] = -1  # walls_length的end 可以用-1代替
        sample_params.update(sample_params_tgt)
        ref_sample_params.update(ref_sample_params_tgt)
        # Add the number of bounding boxes in the scene
        sample_params["length"] = sample_params["class_labels"].shape[0]  # 家具个数不变(不因end_symbol改变)
        sample_params["walls_num"] = walls_num + 1  # 加1是加上索引为[0]的end symbol墙
        return (sample_params, ref_sample_params)

    def collate_fn(self, samples):
        return DatasetCollection.collate_fn(samples)

    @property
    def bbox_dims(self):
        return 7


class AutoregressiveWOCM(Autoregressive):
    '''
    训练数据的制作方法：随机生成一个小于场景家具的index(n_boxes)，以该家具作为gt，gt之前的家具作为输入数据
    '''

    def __init__(self, dataset):
        super().__init__(dataset)
        self.walls_max=dataset.walls_max

    def __getitem__(self, idx):
        params, ref_params = super().__getitem__(idx)
        # Split the boxes and generate input sequences and target boxes
        L, C = params["class_labels"].shape  # L指该场景下的bbx数量(不包括end)，C表示家具类型数量
        n_boxes = np.random.randint(0, L+1)  # 随机生成一个idx作为gt,包括end

        for k, v in params.items():
            # 将原本的参数序列和参数序列_tr，改写为condition和gt
            if "_tr" in k:  # 6项tr：4项物体参数+2项墙参数
                ori_k = k.split("_tr")[0]  # original_k，7项不带tr的原参数
                # 绝对坐标的condition和gt制作
                params[ori_k] = params[ori_k][:n_boxes]  # 输入数据，把n-boxes之前的家具作为输入
                params[k] = v[n_boxes]  # 等价于params[k][n_boxes]  # 将该数据的n-boxes家具作为gt
                # 相对坐标的condition和gt制作
                if not "walls" in ori_k:  # 相对坐标中不需要2项墙参数
                    ref_k = "ref_" + k  # 带tr的
                    ref_ori_k = "ref_" + ori_k  # 不带tr的
                    ref_params[ref_ori_k] = ref_params[ref_ori_k][:, :n_boxes]
                    ref_params[ref_k] = ref_params[ref_k][:, n_boxes]

        params["length"] = n_boxes  # 作为条件的物体数量

        return (params, ref_params, self.walls_max)

def dataset_encoding_factory(
    name,
    dataset,  # raw data, 包含数据的一些基本信息 #训练时是CachedThreedFront
    augmentations=None,  # None
    box_ordering=None  # None
):
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    # note 物体的顺序有可能在数据增强时改变，所以在数据增强后再做物体的序列排列
    if "cached" in name:
        dataset_collection = OrderedDataset(
            CachedDatasetCollection(dataset),  # 输入CachedThreedFront初始化
            ["class_labels", "translations", "sizes", "angles"],  # ordered_keys
            box_ordering=box_ordering  # none
        )
    else:
        box_ordered_dataset = BoxOrderedDataset(  # BoxOrderedDataset
            dataset,
            box_ordering
        )
        room_layout = RoomLayoutEncoder(box_ordered_dataset)
        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)

        dataset_collection = DatasetCollection(
            room_layout,
            class_labels,
            translations,
            sizes,
            angles
        )
    if name == "basic":
        return DatasetCollection(
            class_labels,
            translations,
            sizes,
            angles
        )

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotations":
                print("Applying rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection)
            elif aug_type == "jitter":
                print("Applying jittering augmentations")
                dataset_collection = Jitter(dataset_collection)
    # Scale the input
    dataset_collection = Scale(dataset_collection)  # dataset_collection的类换成Scale的类，只是多了几个scale的函数
    if "eval" in name:
        return dataset_collection
    elif "wocm_no_prm" in name:
        return AutoregressiveWOCM(dataset_collection)
    elif "wocm" in name:  # 训练数据和生成数据
        dataset_collection = Permutation(   # 物体(绝对坐标)序列的顺序排列
            dataset_collection,
            ["class_labels", "translations", "sizes", "angles"]  # permutation_keys
        )
        return AutoregressiveWOCM(dataset_collection)
    else:
        raise NotImplementedError()
