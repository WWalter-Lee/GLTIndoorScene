import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import os
from simple_3dviz.utils import render
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from sklearn.cluster import DBSCAN
from collections import Counter


rough_dir = "../output/rough/"
fine_dir = "../output//fine/"


def check_rough_pred(rough_trans, rough_class, floor_points, wall_points, cluster_para, bound):
    '''
    输入的参数都需要是原尺寸
    :param rough_trans: Tensor: (1,num_box,3)
    :param rough_class: ndarrary: (num_box)
    :param floor_points: ndarrary: (num_wall*2,3)  # 所有墙的点
    :param wall_points: ndarrary: (num_box,2,3)  # 对应墙的点
    '''
    rough_trans = rough_trans.tolist()[0]
    fig = plt.figure()  # 创建一个画板fig
    ax = fig.add_subplot(1, 1, 1)  # 给画板fig添加一个画布ax，画二维的即可
    plt.xlim((-bound - 0.5, bound + 0.5))
    plt.ylim((-bound - 0.5, bound + 0.5))

    # 先用line画floor plan
    ax.plot(floor_points[:,0], -floor_points[:,2])
    # fig.show()
    # 删掉现有目录的所有图
    for f in os.scandir(rough_dir):
        if os.path.isfile(rough_dir + f.name):
            os.remove(rough_dir + f.name)
    for i in range(len(rough_trans)):
        points = rough_trans[i]
        wall_pred = wall_points[i]
        ax.scatter(points[0], -points[2], marker="o")
        ax.plot(wall_pred[:, 0], -wall_pred[:, 2])
        plt.axis("equal")
        plt.axis("off")
        fig.savefig(rough_dir+str(i)+"_"+rough_class[i],dpi=300)

    print("added down:", len(rough_trans))

def check_fine_pred(args,scene,renderables,classes,bbox_params_t):

    for f in os.scandir(fine_dir):
        if os.path.isfile(fine_dir + f.name):
            os.remove(fine_dir + f.name)
    render_check = [renderables[-1]]  # floor_plan
    for i in range(len(renderables) - 1):  # floor在第一个
        render_check.append(renderables[i])
        label = classes[bbox_params_t[0, i + 1, :-7].argmax(-1)]  # start不算
        behaviours_check = [
            LightToCamera(),
            SaveFrames(fine_dir + str(i) + "_" + label + ".png", 360)
        ]
        render(
            renderables=render_check,
            behaviours=behaviours_check,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            n_frames=args.n_frames,
            scene=scene
        )