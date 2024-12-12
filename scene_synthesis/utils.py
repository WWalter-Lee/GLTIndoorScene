# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Modifications Copyright (C) 2024 Yijie Li. All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.

import numpy as np
from PIL import Image
import trimesh

from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh



def get_textured_objects(bbox_params_t, objects_dataset, classes):
    # For each one of the boxes replace them with an object
    renderables = []
    lines_renderables = []
    trimesh_meshes = []
    labels = classes[bbox_params_t[0, :, :-7].argmax(-1)]
    for j in range(1, bbox_params_t.shape[1]):  # 循环bbx个数次
        print(j, "/" ,bbox_params_t.shape[1]-1, ":", labels[j])
        query_size = bbox_params_t[0, j, -4:-1]  # 预测的尺寸
        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]  # 预测的类别
        furniture = objects_dataset.get_closest_furniture_to_box(query_label, query_size)

        # Load the furniture and scale it as it is given in the dataset
        # if furniture == None:
        #     continue
        raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2  # 检索到的模型的中心

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)  # theta弧度制
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
        tr_mesh.visual.material.image = Image.open(
            furniture.texture_image_path
        )
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture)
        )
    )

    return floor, tr_floor

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

def draw_bbx_3d(p_min,p_max,ax):
    # 定义立方体的边界
    x_range = [p_min[0], p_max[0]]
    y_range = [p_min[1], p_max[1]]
    z_range = [p_min[2], p_max[2]]
    # 绘制立方体的边界框
    # 顶上一圈
    ax.plot([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
            [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
            [z_range[0], z_range[0], z_range[0], z_range[0], z_range[0]],
            color="red")
    # 底下一圈
    ax.plot([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
            [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]],
            [z_range[1], z_range[1], z_range[1], z_range[1], z_range[1]],
            color="red")
    #
    ax.plot([x_range[0], x_range[0]],
            [y_range[0], y_range[0]],
            [z_range[0], z_range[1]],
            color='red')
    #
    ax.plot([x_range[1], x_range[1]],
            [y_range[0], y_range[0]],
            [z_range[0], z_range[1]],
            color='red')
    #
    ax.plot([x_range[0], x_range[0]],
            [y_range[1], y_range[1]],
            [z_range[0], z_range[1]],
            color='red')
    #
    ax.plot([x_range[1], x_range[1]],
            [y_range[1], y_range[1]],
            [z_range[0], z_range[1]],
            color='red')

def is_generate_intersec(box, boxes, bounds):
    '''
    检查生成物体是否与现有物体重叠
    输入的都是缩放到[-1,1]的值，所以需要先解缩放
    :param box: 
    :param boxes: 包含start symbole，需去除
    :return: 
    '''
    if boxes["angles"].shape[1] == 1:  # 假如只有start symbol，肯定不重叠
        return False
    # 当前预测box
    if isinstance(box["translations"], int):  #预测eos
        return False
    trans = descale(box["translations"].cpu().numpy(), bounds["translations"][0], bounds["translations"][1])[0][0]
    size = descale(box["sizes"].cpu().numpy(), bounds["sizes"][0], bounds["sizes"][1])[0][0]
    angle = descale(box["angles"].cpu().numpy(), bounds["angles"][0], bounds["angles"][1])[0][0]
    # 八个点先放在原点上
    p1 = np.array([[-size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
    p2 = np.array([[-size[0] / 2], [-size[1] / 2], [size[2] / 2]])
    p3 = np.array([[-size[0] / 2], [size[1] / 2], [-size[2] / 2]])
    p4 = np.array([[-size[0] / 2], [size[1] / 2], [size[2] / 2]])
    p5 = np.array([[size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
    p6 = np.array([[size[0] / 2], [-size[1] / 2], [size[2] / 2]])
    p7 = np.array([[size[0] / 2], [size[1] / 2], [-size[2] / 2]])
    p8 = np.array([[size[0] / 2], [size[1] / 2], [size[2] / 2]])
    p = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8), axis=1)
    # 将八个点旋转
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(angle)
    R[0, 2] = -np.sin(angle)
    R[2, 0] = np.sin(angle)
    R[2, 2] = np.cos(angle)
    R[1, 1] = 1.
    p_new = np.dot(R, p)
    # 找旋转后的bbx边界点
    p_min = (np.min(p_new[0]) + trans[0], np.min(p_new[1]) + trans[1], np.min(p_new[2]) + trans[2])  # 三个维度最小的值
    p_max = (np.max(p_new[0]) + trans[0], np.max(p_new[1]) + trans[1], np.max(p_new[2]) + trans[2])  # 三个维度最大的值

    boxes_trans = descale(boxes["translations"][:,1:,:].detach().cpu().numpy(), bounds["translations"][0], bounds["translations"][1])[0]
    boxes_sizes = descale(boxes["sizes"][:,1:,:].detach().cpu().numpy(), bounds["sizes"][0], bounds["sizes"][1])[0]
    boxes_angles = descale(boxes["angles"][:,1:,:].detach().cpu().numpy(), bounds["angles"][0], bounds["angles"][1])[0]

    is_inter = False  # 初始化为不重叠
    for i in range(boxes_angles.shape[0]):
        trans = boxes_trans[i]
        size = boxes_sizes[i]
        angle = boxes_angles[i]
        # 八个点先放在原点上
        p1 = np.array([[-size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
        p2 = np.array([[-size[0] / 2], [-size[1] / 2], [size[2] / 2]])
        p3 = np.array([[-size[0] / 2], [size[1] / 2], [-size[2] / 2]])
        p4 = np.array([[-size[0] / 2], [size[1] / 2], [size[2] / 2]])
        p5 = np.array([[size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
        p6 = np.array([[size[0] / 2], [-size[1] / 2], [size[2] / 2]])
        p7 = np.array([[size[0] / 2], [size[1] / 2], [-size[2] / 2]])
        p8 = np.array([[size[0] / 2], [size[1] / 2], [size[2] / 2]])
        p = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8), axis=1)
        # 将八个点旋转
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(angle)
        R[0, 2] = -np.sin(angle)
        R[2, 0] = np.sin(angle)
        R[2, 2] = np.cos(angle)
        R[1, 1] = 1.
        bbx_new = np.dot(R, p)
        # 找旋转后的bbx边界点
        bbx_min = (np.min(bbx_new[0]) + trans[0], np.min(bbx_new[1]) + trans[1], np.min(bbx_new[2]) + trans[2])  # 三个维度最小的值
        bbx_max = (np.max(bbx_new[0]) + trans[0], np.max(bbx_new[1]) + trans[1], np.max(bbx_new[2]) + trans[2])  # 三个维度最大的值
        # 检查的方法就是算两个bbx的边界点，小边界点中取大的值，大边界点中取小的值，看是否有交集
        x1 = max(p_min[0], bbx_min[0])
        y1 = max(p_min[1], bbx_min[1])
        z1 = max(p_min[2], bbx_min[2])
        x2 = min(p_max[0], bbx_max[0])
        y2 = min(p_max[1], bbx_max[1])
        z2 = min(p_max[2], bbx_max[2])
        # 重叠检测,大边界的取的点的值都要比小边界取的点大即为有重叠
        if x2 >= x1 and y2 >= y1 and z2 >= z1:
            is_inter = True
            break
        else:
            continue
    return is_inter

def check_generate_intersec(box, boxes, bounds, ax):
    '''
    检查生成物体是否与现有物体重叠，绘图，所以需要传入ax
    输入的都是缩放到[-1,1]的值，所以需要先解缩放
    :param box:
    :param boxes: 包含start symbole，需去除
    :return:
    '''
    if boxes["angles"].shape[1] == 1:  # 假如只有start symbol，肯定不重叠
        return False
    # 当前预测box
    trans = descale(box["translations"].detach().cpu().numpy(), bounds["translations"][0], bounds["translations"][1])[0][0]
    size = descale(box["sizes"].detach().cpu().numpy(), bounds["sizes"][0], bounds["sizes"][1])[0][0]
    angle = descale(box["angles"].detach().cpu().numpy(), bounds["angles"][0], bounds["angles"][1])[0][0]
    # 八个点先放在原点上
    p1 = np.array([[-size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
    p2 = np.array([[-size[0] / 2], [-size[1] / 2], [size[2] / 2]])
    p3 = np.array([[-size[0] / 2], [size[1] / 2], [-size[2] / 2]])
    p4 = np.array([[-size[0] / 2], [size[1] / 2], [size[2] / 2]])
    p5 = np.array([[size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
    p6 = np.array([[size[0] / 2], [-size[1] / 2], [size[2] / 2]])
    p7 = np.array([[size[0] / 2], [size[1] / 2], [-size[2] / 2]])
    p8 = np.array([[size[0] / 2], [size[1] / 2], [size[2] / 2]])
    p = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8), axis=1)
    # 将八个点旋转
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(angle)
    R[0, 2] = -np.sin(angle)
    R[2, 0] = np.sin(angle)
    R[2, 2] = np.cos(angle)
    R[1, 1] = 1.
    p_new = np.dot(R, p)
    # 找旋转后的bbx边界点
    p_min = (np.min(p_new[0]) + trans[0], np.min(p_new[1]) + trans[1], np.min(p_new[2]) + trans[2])  # 三个维度最小的值
    p_max = (np.max(p_new[0]) + trans[0], np.max(p_new[1]) + trans[1], np.max(p_new[2]) + trans[2])  # 三个维度最大的值
    # 绘出bbx
    draw_bbx_3d(p_min, p_max, ax)  # 只留下两个变量p_min,p_max

    boxes_trans = descale(boxes["translations"][:,1:,:].detach().cpu().numpy(), bounds["translations"][0], bounds["translations"][1])[0]
    boxes_sizes = descale(boxes["sizes"][:,1:,:].detach().cpu().numpy(), bounds["sizes"][0], bounds["sizes"][1])[0]
    boxes_angles = descale(boxes["angles"][:,1:,:].detach().cpu().numpy(), bounds["angles"][0], bounds["angles"][1])[0]

    is_inter = False  # 初始化为不重叠
    for i in range(boxes_angles.shape[0]):
        trans = boxes_trans[i]
        size = boxes_sizes[i]
        angle = boxes_angles[i]
        # 八个点先放在原点上
        p1 = np.array([[-size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
        p2 = np.array([[-size[0] / 2], [-size[1] / 2], [size[2] / 2]])
        p3 = np.array([[-size[0] / 2], [size[1] / 2], [-size[2] / 2]])
        p4 = np.array([[-size[0] / 2], [size[1] / 2], [size[2] / 2]])
        p5 = np.array([[size[0] / 2], [-size[1] / 2], [-size[2] / 2]])
        p6 = np.array([[size[0] / 2], [-size[1] / 2], [size[2] / 2]])
        p7 = np.array([[size[0] / 2], [size[1] / 2], [-size[2] / 2]])
        p8 = np.array([[size[0] / 2], [size[1] / 2], [size[2] / 2]])
        p = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8), axis=1)
        # 将八个点旋转
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(angle)
        R[0, 2] = -np.sin(angle)
        R[2, 0] = np.sin(angle)
        R[2, 2] = np.cos(angle)
        R[1, 1] = 1.
        bbx_new = np.dot(R, p)
        # 找旋转后的bbx边界点
        bbx_min = (np.min(bbx_new[0]) + trans[0], np.min(bbx_new[1]) + trans[1], np.min(bbx_new[2]) + trans[2])  # 三个维度最小的值
        bbx_max = (np.max(bbx_new[0]) + trans[0], np.max(bbx_new[1]) + trans[1], np.max(bbx_new[2]) + trans[2])  # 三个维度最大的值
        # 绘出bbx
        draw_bbx_3d(bbx_min, bbx_max, ax)
        # 检查的方法就是算两个bbx的边界点，小边界点中取大的值，大边界点中取小的值，看是否有交集
        x1 = max(p_min[0], bbx_min[0])
        y1 = max(p_min[1], bbx_min[1])
        z1 = max(p_min[2], bbx_min[2])
        x2 = min(p_max[0], bbx_max[0])
        y2 = min(p_max[1], bbx_max[1])
        z2 = min(p_max[2], bbx_max[2])
        # 重叠检测,大边界的取的点的值都要比小边界取的点大即为有重叠
        if x2 >= x1 and y2 >= y1 and z2 >= z1:
            is_inter = True
            break
        else:
            continue
    return is_inter
