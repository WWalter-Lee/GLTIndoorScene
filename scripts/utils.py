# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#
import copy
import os

import numpy as np
import torch
from PIL import Image
from pyrr import Matrix44
import sys
import trimesh

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.utils import save_frame
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render as render_simple_3dviz

sys.path.append("..")
from scene_synthesis.utils import get_textured_objects
# from simple_3dviz.window import show
import matplotlib.pyplot as plt
import shutil


class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def ensure_parent_directory_exists(filepath):
    os.makedirs(filepath, exist_ok=True)


def floor_plan_renderable(room, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces, flg = room.floor_plan
    # Center the floor
    vertices -= room.floor_plan_centroid  # 将floorplan的中心位置移到坐标原点
    # Return a simple-3dviz renderable
    # a = Mesh.from_faces(vertices, faces, color)
    # show(a)
    return Mesh.from_faces(vertices, faces, color)


def floor_plan_from_scene(
    scene,
    path_to_floor_plan_textures,
    without_room_mask=False  # without:false即为要room mask
):
    if not without_room_mask:  # 有room mask
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))  # (64,64,1)->(1,1,64,64)
        )
    else:                      # 无room mask(新数据)
        room_mask = None
    # Also get a renderable for the floor plan
    floor, tr_floor = get_floor_plan(
        scene,
        [
            os.path.join(path_to_floor_plan_textures, fi)
            for fi in os.listdir(path_to_floor_plan_textures)
        ]
    )
    return [floor], [tr_floor], room_mask


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces, flg = scene.floor_plan  # (30,3),(10,3)
    vertices = vertices - scene.floor_plan_centroid  # floor_plan_centroid: (3)
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(  # 利用顶点等信息创建TexturedMesh floor
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(  # Trimesh
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture)
        )
    )
    return floor, tr_floor


def get_textured_objects_in_scene(scene, ignore_lamps=False):
    renderables = []
    for furniture in scene.bboxes:
        model_path = furniture.raw_model_path
        if not model_path.endswith("obj"):
            import pdb
            pdb.set_trace()

        # Load the furniture and scale it as it is given in the dataset
        raw_mesh = TexturedMesh.from_file(model_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)
    return renderables


def render(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()  # Scene
    for r, c in zip(renderables, color):  # r:Mesh
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode  # shading 变 flat
            r.colors = c
        scene.add(r)
    scene.render()
    # test = scene.frame[:,:,0]
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )
    return scene


def export_scene(output_directory, trimesh_meshes, names=None):
    if names is None:
        names = [
            "object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))
        ]
    mtl_names = [
        "material_{:03d}".format(i) for i in range(len(trimesh_meshes))
    ]

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(
            m,
            return_texture=True
        )

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(
                    b"material0", mtl_names[i].encode("ascii")
                )
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def print_predicted_labels(dataset, boxes):
    object_types = np.array(dataset.object_types)
    box_id = boxes["class_labels"][0, 1:].argmax(-1)
    labels = object_types[box_id.cpu().numpy()].tolist()
    print("The predicted scene contains {}".format(labels))


def poll_specific_class(dataset):
    label = input(
        "Select an object class from {}\n".format(dataset.object_types)
    )
    if label in dataset.object_types:
        return dataset.object_types.index(label)
    else:
        return None


def make_network_input(current_boxes, indices):
    def _prepare(x):
        return torch.from_numpy(x[None].astype(np.float32))

    return dict(
        class_labels=_prepare(current_boxes["class_labels"][indices]),
        translations=_prepare(current_boxes["translations"][indices]),
        sizes=_prepare(current_boxes["sizes"][indices]),
        angles=_prepare(current_boxes["angles"][indices])
    )


def render_to_folder(
    args,
    folder,
    dataset,
    objects_dataset,
    tr_floor,
    floor_plan,
    scene,
    bbox_params,
    add_start_end=False
):
    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu()

    if add_start_end:
        bbox_params_t = torch.cat([
            torch.zeros(1, 1, bbox_params_t.shape[2]),
            bbox_params_t,
            torch.zeros(1, 1, bbox_params_t.shape[2]),
        ], dim=1)

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels)
    )
    trimesh_meshes += tr_floor

    path_to_objs = os.path.join(args.output_directory, folder)
    if not os.path.exists(path_to_objs):
        os.mkdir(path_to_objs)
    export_scene(path_to_objs, trimesh_meshes)

    path_to_image = os.path.join(
        args.output_directory,
        folder + "_render.png"
    )
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image, 1)
    ]
    render_simple_3dviz(
        renderables + floor_plan,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )


def render_scene_from_bbox_params(
    args,
    bbox_params,
    dataset,
    objects_dataset,
    classes,
    floor_plan,
    tr_floor,
    scene,
    path_to_image,
    path_to_objs
):
    boxes = dataset.post_process(bbox_params)
    print_predicted_labels(dataset, boxes)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu().numpy()

    renderables, trimesh_meshes = get_textured_objects(
        bbox_params_t, objects_dataset, classes
    )
    renderables += floor_plan
    trimesh_meshes += tr_floor

    # Do the rendering
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image+".png", 1)
    ]
    render_simple_3dviz(
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
    if trimesh_meshes is not None:
        # Create a trimesh scene and export it
        if not os.path.exists(path_to_objs):
            os.mkdir(path_to_objs)
        else:  # 如果已经有obj，要先删光
            shutil.rmtree(path_to_objs)
            os.makedirs(path_to_objs)
        export_scene(path_to_objs, trimesh_meshes)

def scale_walls(walls_ori, bounds):
    walls = copy.deepcopy(walls_ori)  # 拷贝，因为原值还有用
    points = np.stack(walls["points"])
    # points_m = np.sum(points, axis=1, keepdims=True)/2
    length = np.stack(walls["length"])
    origin = walls["origin"]

    # 用bounds对墙的参数缩放到[-1,1]
    walls["points"] = scale(points, bounds["translations"][0],bounds["translations"][1])
    walls["length"] = scale(length, 0, bounds["length"])
    walls["origin_scaled"] = scale(origin, bounds["translations"][0],bounds["translations"][1])
    return walls
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

def check_gt_bbx(bbox_params_t, floor_points):
    '''
    检查并绘制物体bbx
    '''
    fig = plt.figure()  # 创建一个画板fig

    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    # 固定相机视角
    ax.view_init(elev=0, azim=90, roll=0)
    # 先用line画floor plan
    ax.plot(floor_points[:, 0], floor_points[:, 1], floor_points[:, 2], label="floor_plan")
    fig.show()

    trans = bbox_params_t[:,:,-7:-4][0]
    sizes = bbox_params_t[:,:,-4:-1][0]
    angles = bbox_params_t[:,:,-1][0]

    bbxes = []  # 之前生成的
    for i in range(len(angles)):
        center = trans[i]
        size = sizes[i]
        angle = angles[i]
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
        p_min = (np.min(p_new[0])+center[0], np.min(p_new[1])+center[1], np.min(p_new[2])+center[2])  # 三个维度最小的值
        p_max = (np.max(p_new[0])+center[0], np.max(p_new[1])+center[1], np.max(p_new[2])+center[2])  # 三个维度最大的值
        # 绘出bbx
        draw_bbx_3d(p_min, p_max, ax)

        # 检查是否有重叠,当前的生成的p_min,p_max和之前已有的bbxes
        if len(bbxes) >= 1:
            for j in range(len(bbxes)):
                # 已有的bbx
                bbx_min = bbxes[j][0]
                bbx_max = bbxes[j][1]
                # 检查的方法就是算两个bbx的边界点，小边界点中取大的值，大边界点中取小的值，看是否有交集
                x1 = max(p_min[0],bbx_min[0])
                y1 = max(p_min[1], bbx_min[1])
                z1 = max(p_min[2], bbx_min[2])
                x2 = min(p_max[0], bbx_max[0])
                y2 = min(p_max[1], bbx_max[1])
                z2 = min(p_max[2], bbx_max[2])
                # 计算交集体积
                # v = (x2-x1)*(y2-y1)*(z2-z1)  # 不用计算面积，只需要比较值
                if x2>x1 and y2>y1 and z2>z1:  # 面积大于0且大边界的取的点的值都要比小边界取的点大
                    is_intersect = True  # 表示有重叠
                else:
                    is_intersect = False
                print(is_intersect)
        bbxes.append((p_min,p_max))

    # print(1)

def distance_point_to_line(P, A, B):
    '''
    计算点P到线段AB的距离
    '''
    AP = A-P  # 向量AP
    AB = B-A  # 向量AB

    # 叉乘计算面积
    S = np.linalg.norm(np.cross(AP,AB))  # |AP||AB|sin(a)，值为向量的模长

    AB_len = np.linalg.norm(AB)  # |AB|

    # 按理来讲AB_len应该不会为0才对，可能还是简化墙边界的时候没简化好？
    dis = S/AB_len  # |AP|sin(a)

    return dis