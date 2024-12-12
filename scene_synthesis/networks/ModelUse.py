#
# Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

import torch
import torch.nn as nn
import numpy as np

torch.autograd.set_detect_anomaly(True)

from ..datasets.global_ref_transform import global2ref,ref2global
from ..datasets.utils import scale, descale
from ..utils import is_generate_intersec
import matplotlib.pyplot as plt


class ModelUse():
    '''
    use rough and fine models
    '''
    def __init__(self, rough_model, fine_model):
        self.rough_model = rough_model
        self.fine_model = fine_model
        self.n_classes = self.rough_model.n_classes
        self.max_walls = self.fine_model.walls_max
    def start_symbol(self, device="cpu"):

        start_class = torch.zeros(1, 1, self.n_classes, device=device)  # (1,1,23)
        start_class[0, 0, -2] = 1  # 将倒数第二个0变为1，表示start标识
        return {
            "class_labels": start_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device),
            "walls_order": torch.zeros(1, 1, self.max_walls, device=device),  # batch数, 物体数，以及物体对应的墙索引
            "walls_length": torch.zeros(1, 1, 1, device=device),  # batch数, 物体数，以及墙索引对应的墙长度
        }

    def autoregressive_decode(self, boxes, walls, room_mask, classes, cluster, combs):
        '''
        生成单个物体
        :param boxes: 现有box的信息
        :param walls: 墙的信息
        :param room_mask: room边界
        :param classes: 该数据集所有的家具类别，从config读取
        :param cluster: 该数据集的聚类信息，从config读取
        :param combs: 该场景维护的组合，在函数中更改.list{num_combs.dict{"lines","wall_idx"}}
        :return: 自回归生成的rough信息和fine信息，还有要更新的comb信息
        '''

        rough_info = self.rough_model._encode_rough(boxes, walls, room_mask) #rough中的wall_pred是已经加了1的，不会是[0]墙

        comb = {}  # 假如没有要更新的就将空dict返回
        # 用rough_class来判定结束
        rough_class = rough_info["rough_class"]
        if rough_class[0, 0, -1] == 1:  # eos 物体
            return (
                rough_info,
                {
                    "class_labels": rough_class,
                    "translations": 0,
                    "sizes": 0,
                    "angles": 1
                },
                comb
            )

        # 在rough和fine模型之间，判断是否有常见组
        label = classes[np.argmax(rough_info["rough_class"].detach().cpu().numpy(), -1)].item()
        clusters = [dict(c) for c in cluster]  # 将cluster从tuple转换为dict
        # 思想: 为场景维护若干combs_list，每个comb由dict{"lines","wall_idx"}两项构成,comb在函数外面的重叠检测之后确定了box之后更新(维护)
        # 先查看当前生成的label是否在cluster中
        in_clusters = False
        for cluster in clusters:
            if label in cluster:
                in_clusters = True
                break
        # 如果生成的物体在cluster中，则到场景的combs中去找能否加到其中的comb中
        candidates = []  # 如果生成的物体能加到多个comb中，则找距离最近的墙(comb)加入
        if in_clusters == True:
            comb_idx = -1
            for comb in combs:  # 每个comb由dict{"lines","wall_idx"}两项构成
                # 检查每个comb的每个组合，看box是否在comb中
                comb_idx += 1  #
                for line in comb["lines"]:
                    if label in line:
                        candidates.append({"wall_idx": comb["wall_idx"], "comb_idx": comb_idx})  # 存入墙索引
                        break  # 有一个符合即可退出单个comb循环，查其他comb
            # 如果该物体能加入某些comb，则计算该物体到多个comb的距离z
            if not len(candidates) == 0:
                dis_min=9999
                comb_idx = None
                for candi in candidates:
                    if isinstance(candi["wall_idx"],int):
                        wall_idx=candi["wall_idx"]
                    else:
                        wall_idx = candi["wall_idx"].cpu().item()
                    wall_center = walls["origin_scaled"][wall_idx-1] # 只在这里取walls的点时需要-1
                    dis = np.sum((rough_info["rough_trans"].cpu().numpy()[0][0]-wall_center)**2)
                    if dis<dis_min:
                        dis_min=dis
                        comb_idx=candi["comb_idx"]
                comb = combs[comb_idx]
            # 如果该物体没能加入任何comb，则创建comb，初始comb就是config中统计的list
            else:
                # comb = {"lines": clusters, "wall_idx": rough_info["wall_pred"]}  # 这里的wall_pred对应的是正常wall+1,所以应该-1，才能在walls中找到正确的墙
                # 检查该墙有无分配常见组，如果有，则随机选另一面长度较长的墙；直接抽取现有comb已分配的墙
                walls_current = []  # 现有墙的集合
                for c in combs:
                    walls_current.append(c["wall_idx"])
                walls_current = torch.tensor(walls_current,device=rough_class.device)
                if torch.isin(rough_info["wall_pred"], walls_current):
                    walls_idx = torch.arange(len(walls["length"]))
                    walls_choose = np.setdiff1d(walls_idx.cpu(), walls_current.cpu())
                    if not len(walls_choose) == 0:
                        rough_info["wall_pred"] = torch.tensor(np.random.choice(walls_choose),
                                                           device=rough_class.device,dtype=torch.int64).reshape(1, 1)
                # 这里创建的comb不能包含与该物体无关的
                comb = {"lines": [], "wall_idx": rough_info["wall_pred"]}
                for line in clusters:
                    if label in line:
                        comb["lines"].append(line)
                combs.append(comb)
            # 最终要更新该物体对应的墙索引
            if isinstance(comb["wall_idx"],int):
                rough_info["wall_pred"] = torch.tensor(comb["wall_idx"]).reshape(1, 1)
            else:
                rough_info["wall_pred"] = comb["wall_idx"]

        wall_basis, origin_scaled, F = self.fine_model._encode_fine(boxes, walls, rough_info)  # (1,1,512) 结合现有boxes+walls # 预测的q^

        class_labels = self.fine_model.hidden2fine.sample_class_labels(F)  # (1,1,23)
        # Sample the translations
        translations_ref = self.fine_model.hidden2fine.sample_translations(F, class_labels)  # (1,1,3),这里预测的是相对坐标，需要加回墙的坐标
        # 将局部坐标转换为全局坐标
        translations_ref = translations_ref.cpu().numpy().reshape(-1, 3)
        translations = ref2global(wall_basis.T, origin_scaled, translations_ref)
        translations_ref = torch.from_numpy(translations_ref).float().to(class_labels.device)[None, :, :]
        translations = torch.from_numpy(translations).float().to(class_labels.device)[None, :, :]  # 转换成float32才能和后续统一
        # Sample the angles
        angles_ref = self.fine_model.hidden2fine.sample_angles(  # (1,1,1)
            F, class_labels, translations_ref
        )
        # # 将局部角度转换为全局角度
        angle_min = - np.pi
        angle_max = np.pi
        wall_basis_x = wall_basis[0]
        x = np.array([1, 0, 0])
        cos_theta = np.dot(wall_basis_x, x)  # 这里顺序无关
        theta = np.arccos(cos_theta)         # 要旋转的角度
        # 原本是给定全局角度，加上要旋转的角即为局部角；现在为给定局部角，减去要旋转的角即为全局角
        angles_ref = angles_ref.cpu().numpy().reshape(-1, 1)  # 局部角度
        # 先将angles_ref解缩放回真实值
        ref_ori = descale(angles_ref, angle_min, angle_max)
        # 旋转方向
        ore = np.cross(wall_basis_x, x)  #假如为正，说明x到wall_basis为顺时针，但现在是从wall到x，所以为正时要取-
        if ore[1] >= 0:
            theta = -theta

        # 将真实值旋转
        angles = (ref_ori + theta - angle_min) % (2 * np.pi) + angle_min
        # 将真实值缩放回[-1,1],之后和坐标等统一再解缩放
        angles = scale(angles, angle_min, angle_max)

        angles_ref = torch.from_numpy(angles_ref).float().to(class_labels.device)[None, :, :]
        angles = torch.from_numpy(angles).float().to(class_labels.device)[None, :, :]
        # Sample the sizes
        sizes = self.fine_model.hidden2fine.sample_sizes(  # (1,1,3)
            F, class_labels, translations_ref, angles_ref
        )
        # 本次自回归生成的物体
        box = {"class_labels": class_labels, "translations": translations, "sizes": sizes, "angles": angles,
               "walls_order": torch.nn.functional.one_hot(rough_info["wall_pred"], num_classes=self.max_walls)}

        return (rough_info, box, comb)

    @torch.no_grad()
    def generate_boxes(self, walls, room_mask, bounds, floor_points, classes,
            cluster, max_boxes=32, device="cpu"):
        '''
        :param walls: 给的是缩放后的信息，因为要和模型训练时保持一致
        :param floor_points: 可视化重叠检测要用
        '''
        boxes = self.start_symbol(device)  # 从起始标识符开始生成 #训练的时候没有,并且在推断时候也是累赘的这么一个symbol
        rough_class = []
        rough_trans = []
        wall_pred = []
        # 推断时要维护的comb信息
        combs = []
        for i in range(max_boxes):  # 32
            rough_info, box, comb = self.autoregressive_decode(boxes, walls=walls, room_mask=room_mask, classes=classes,
            cluster=cluster, combs=combs)  # 预测下一个box
            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:  # 如果预测出end object，退出循环
                break

            check_intersec = is_generate_intersec(box, boxes, bounds)  # 实际使用重叠检测
            # check_intersec = False # 不做重叠检测

            # 如果生成的是椅子，不做重叠检测
            box_label = classes[np.argmax(box["class_labels"].detach().cpu().numpy(), -1).item()]
            if "chair" in box_label:
                check_intersec = False

            for time in range(10):  # 最多重叠检测的次数
                if check_intersec == False:
                    break
                else:
                    # 重新生成
                    rough_info, box, comb = self.autoregressive_decode(boxes, walls=walls, room_mask=room_mask,classes=classes,cluster = cluster, combs=combs)
                    check_intersec = is_generate_intersec(box, boxes, bounds)

            if box["class_labels"][0, 0, -1] == 1:  # 如果预测出end object，退出循环
                break

            # 确定box的信息后再更新comb (当前label对应的value-1，为0时删掉该key)
            if not len(comb) == 0:
                label = classes[np.argmax(box["class_labels"].cpu().numpy(), -1)].item()
                for line in comb["lines"]:
                    if label in line:
                        line[label] -= 1
                        if line[label] == 0:
                            del line[label]

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)  # 将预测的box加入到现有物体序列中
            rough_class.append(rough_info["rough_class"])
            rough_trans.append(rough_info["rough_trans"])
            wall_pred.append(rough_info["wall_pred"])

        rough_info = {"rough_class": rough_class, "rough_trans": rough_trans, "wall_pred": wall_pred}
        return (
            rough_info,
            {
                "class_labels": boxes["class_labels"].to("cpu"),
                "translations": boxes["translations"].to("cpu"),
                "sizes": boxes["sizes"].to("cpu"),
                "angles": boxes["angles"].to("cpu")
            }
        )