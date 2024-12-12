#
# Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
#

import torch
from .base import FixedPositionalEncoding, sample_from_dmll
from ..datasets.global_ref_transform import global2ref,ref2global_tensor
from ..datasets.utils import scale, descale
import os,json
from ..stats_logger import StatsLogger

def train_on_batch_rough(model_rough, optimizer_rough, sample_params, config):
    # Make sure that everything has the correct size
    optimizer_rough.zero_grad()

    X_rough_pred = model_rough(sample_params)  # model:AutoregressiveTransformer, X_pred: AutoregressiveBBoxOutput# 每个预测值（除了类别）的最后一维是30维，表示10个分布

    loss_rough = X_rough_pred.reconstruction_loss_rough(sample_params)
    # Do the backpropagation
    loss_rough.backward()
    # Do the update
    optimizer_rough.step()

    return loss_rough.item()


@torch.no_grad()
def validate_on_batch_rough(model_rough, sample_params, config):
    X_rough_pred = model_rough(sample_params)  # model:AutoregressiveTransformer, X_pred: AutoregressiveBBoxOutput# 每个预测值（除了类别）的最后一维是30维，表示10个分布
    loss_rough = X_rough_pred.reconstruction_loss_rough(sample_params)
    return loss_rough.item()

def train_on_batch_fine(model_fine, optimizer_fine, sample_params, config):
    # Make sure that everything has the correct size
    optimizer_fine.zero_grad()

    X_fine_pred = model_fine(sample_params)  # model:AutoregressiveTransformer, X_pred: AutoregressiveBBoxOutput# 每个预测值（除了类别）的最后一维是30维，表示10个分布
    loss_fine = X_fine_pred.reconstruction_loss_fine(sample_params)
    loss_fine.backward()  # 不加人体工学损失训练

    # Do the update
    optimizer_fine.step()

    return loss_fine.item()


@torch.no_grad()
def validate_on_batch_fine(model_fine, sample_params, config):
    X_fine_pred = model_fine(sample_params)  # model:AutoregressiveTransformer, X_pred: AutoregressiveBBoxOutput# 每个预测值（除了类别）的最后一维是30维，表示10个分布
    loss_fine = X_fine_pred.reconstruction_loss_fine(sample_params)

    return loss_fine.item()
#
# def compute_ergoloss(X_fine_pred,sample_params,config):
#     device = sample_params["room_layout"].device
#     # 先采样得到预测的相对坐标
#     B = X_fine_pred.class_labels.shape[0]
#     t_x = sample_from_dmll(X_fine_pred.translations_x.reshape(B, -1))
#     t_y = sample_from_dmll(X_fine_pred.translations_y.reshape(B, -1))
#     t_z = sample_from_dmll(X_fine_pred.translations_z.reshape(B, -1))
#     trans_ref = torch.cat([t_x, t_y, t_z], dim=-1).view(B, 3)
#     # angles_ref = sample_from_dmll(X_fine_pred.angles.reshape(B, -1))
#     # 读取bounds_translations
#     with open('../data/bedroom/dataset_stats.txt', 'r') as file:
#         json_data = file.read()
#     data = json.loads(json_data)
#     bounds = list(map(float, data.get('bounds_translations', [])))
#     bounds = (torch.tensor((bounds[:3]),device=device),torch.tensor(bounds[3:],device=device))
#     class_labels = data["class_labels"]
#
#     count = 0  # 统计需要计算人体工学的场景个数
#     ergo_loss = torch.tensor(0.0, device=device)
#     for i in range(B):  # 只能单个场景转换并计算
#         label = class_labels[X_fine_pred.class_labels[i].argmax(-1)]
#         idx = -1  # 找已有物体中是否有匹配项
#         if label == "sofa" or label=="tv_stand":  # 预测的是sofa
#             labels_idx = sample_params["class_labels"][i].argmax(-1)
#             current_labels = [class_labels[idx] for idx in labels_idx]
#             if label == "sofa" and "tv_stand" in current_labels:
#                 idx = current_labels.index("tv_stand")
#             elif label=="tv_stand" and "sofa" in current_labels:
#                 idx = current_labels.index("sofa")
#         if idx == -1:  # 没有匹配项
#             continue
#         # 开始计算los！
#         # 不需要预测角度
#         # angle_ref = angles_ref[i]
#         # tran, angle = ref2global(sample_params, tran_ref, angle_ref,i,device, bounds)  # 得到预测家具的真实坐标和角度
#         tran_ref = trans_ref[i]
#         tran = ref2global(sample_params, tran_ref, i, bounds)
#         tran_tgt = descale(sample_params["translations"][i][idx],bounds[0], bounds[1])
#         wall_idx = int(sample_params["walls_order_tr"][i])
#         wall_basis = sample_params["walls_basis"][i][wall_idx] # 墙的顺序是顺时针方向
#         # los计算：trans指向tgt的向量，与angle本身的角度向量的cos
#         # 直接通过预测对应墙的向量作为预测物体的方向向量，避免方向转换错误
#         v = (tran_tgt-tran)[0][:2] # 预测物体指向目标物体的向量 # 丢弃z坐标
#         u = wall_basis[0][:2]  # x轴的基 丢弃z坐标
#         # 直接计算单位向量点乘(u,v)作为los，越接近垂直，值约接近于0
#         los = torch.dot(v/torch.norm(v),u) * torch.dot(v/torch.norm(v),u)  # 点乘的平方，考虑正负值
#         ergo_loss += los
#         count += 1
#     if count != 0:
#         ergo_loss = ergo_loss/count
#     return ergo_loss
def ref2global(sample_params,tran_ref,i,bounds):
    # 将相对坐标转换回绝对坐标
    wall_idx = int(sample_params["walls_order_tr"][i].item())
    wall_basis = sample_params["walls_basis"][i][wall_idx]
    origin_scaled = sample_params["walls_origin_scaled"][i][wall_idx]
    tran = ref2global_tensor(wall_basis.T, origin_scaled, tran_ref[None, :])  # 输入的都是缩放后的数值
    tran = descale(tran, bounds[0], bounds[1])  # 解缩放回真实值

    return tran
