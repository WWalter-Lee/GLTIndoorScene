import torch
import torch.nn as nn
import numpy as np

torch.autograd.set_detect_anomaly(True)

from ..datasets.global_ref_transform import global2ref,ref2global
from ..datasets.utils import scale, descale
from ..utils import is_generate_intersec
import matplotlib.pyplot as plt


class ModelUse():
    def __init__(self, rough_model, fine_model):
        self.rough_model = rough_model
        self.fine_model = fine_model
        self.n_classes = self.rough_model.n_classes
        self.max_walls = self.fine_model.walls_max
    def start_symbol(self, device="cpu"):

        start_class = torch.zeros(1, 1, self.n_classes, device=device)  #
        start_class[0, 0, -2] = 1  #
        return {
            "class_labels": start_class,
            "translations": torch.zeros(1, 1, 3, device=device),
            "sizes": torch.zeros(1, 1, 3, device=device),
            "angles": torch.zeros(1, 1, 1, device=device),
            "walls_order": torch.zeros(1, 1, self.max_walls, device=device),  #
            "walls_length": torch.zeros(1, 1, 1, device=device),  #
        }
        # return boxes

    def autoregressive_decode(self, boxes, walls, room_mask, classes, cluster, combs):
        rough_info = self.rough_model._encode_rough(boxes, walls, room_mask) #

        comb = {}  #
        rough_class = rough_info["rough_class"]
        if rough_class[0, 0, -1] == 1:  # eos
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

        #
        label = classes[np.argmax(rough_info["rough_class"].detach().cpu().numpy(), -1)].item()
        clusters = [dict(c) for c in cluster]  #
        #
        in_clusters = False
        for cluster in clusters:
            if label in cluster:
                in_clusters = True
                break
        #
        candidates = []  #
        if in_clusters == True:
            comb_idx = -1
            for comb in combs:  #
                #
                comb_idx += 1  #
                for line in comb["lines"]:
                    if label in line:
                        candidates.append({"wall_idx": comb["wall_idx"], "comb_idx": comb_idx})  #
                        break  #
            #
            if not len(candidates) == 0:
                dis_min=9999
                comb_idx = None
                for candi in candidates:
                    if isinstance(candi["wall_idx"],int):
                        wall_idx=candi["wall_idx"]
                    else:
                        wall_idx = candi["wall_idx"].cpu().item()
                    wall_center = walls["origin_scaled"][wall_idx-1] #
                    dis = np.sum((rough_info["rough_trans"].cpu().numpy()[0][0]-wall_center)**2)
                    if dis<dis_min:
                        dis_min=dis
                        comb_idx=candi["comb_idx"]
                #
                comb = combs[comb_idx]
            #
            else:
                walls_current=[] #
                for c in combs:
                    walls_current.append(c["wall_idx"])
                walls_current = torch.tensor(walls_current,device=rough_class.device)
                if torch.isin(rough_info["wall_pred"], walls_current):
                    walls_idx = torch.arange(len(walls["length"]))
                    walls_choose = np.setdiff1d(walls_idx.cpu(), walls_current.cpu())
                    if not len(walls_choose) == 0:
                        rough_info["wall_pred"] = torch.tensor(np.random.choice(walls_choose),
                                                           device=rough_class.device,dtype=torch.int64).reshape(1, 1)
                #
                comb = {"lines": [], "wall_idx": rough_info["wall_pred"]}
                for line in clusters:
                    if label in line:
                        comb["lines"].append(line)
                combs.append(comb)
            # #
            if isinstance(comb["wall_idx"],int):
                rough_info["wall_pred"] = torch.tensor(comb["wall_idx"]).reshape(1, 1)
            else:
                rough_info["wall_pred"] = comb["wall_idx"]

        wall_basis, origin_scaled, F = self.fine_model._encode_fine(boxes, walls, rough_info)  #

        class_labels = self.fine_model.hidden2fine.sample_class_labels(F)  #
        # Sample the translations
        translations_ref = self.fine_model.hidden2fine.sample_translations(F, class_labels)  #
        #
        translations_ref = translations_ref.cpu().numpy().reshape(-1, 3)
        translations = ref2global(wall_basis.T, origin_scaled, translations_ref)
        translations_ref = torch.from_numpy(translations_ref).float().to(class_labels.device)[None, :, :]
        translations = torch.from_numpy(translations).float().to(class_labels.device)[None, :, :]  #
        # Sample the angles
        angles_ref = self.fine_model.hidden2fine.sample_angles(  # (1,1,1)
            F, class_labels, translations_ref
        )
        # #
        angle_min = - np.pi
        angle_max = np.pi
        wall_basis_x = wall_basis[0]
        x = np.array([1, 0, 0])
        cos_theta = np.dot(wall_basis_x, x)  #
        theta = np.arccos(cos_theta)  #
        #
        angles_ref = angles_ref.cpu().numpy().reshape(-1, 1)  #
        #
        ref_ori = descale(angles_ref, angle_min, angle_max)
        #
        ore = np.cross(wall_basis_x, x)  #
        if ore[1] >= 0:
            theta = -theta

        #
        angles = (ref_ori + theta - angle_min) % (2 * np.pi) + angle_min
        #
        angles = scale(angles, angle_min, angle_max)

        angles_ref = torch.from_numpy(angles_ref).float().to(class_labels.device)[None, :, :]
        angles = torch.from_numpy(angles).float().to(class_labels.device)[None, :, :]
        # Sample the sizes
        sizes = self.fine_model.hidden2fine.sample_sizes(  # (1,1,3)
            F, class_labels, translations_ref, angles_ref
        )
        #
        box = {"class_labels": class_labels, "translations": translations, "sizes": sizes, "angles": angles,
               "walls_order": torch.nn.functional.one_hot(rough_info["wall_pred"], num_classes=self.max_walls)}

        return (rough_info, box, comb)
    @torch.no_grad()
    def generate_boxes(self, walls, room_mask, bounds, floor_points, classes,
            cluster, max_boxes=32, device="cpu"):
        boxes = self.start_symbol(device)  #
        rough_class = []
        rough_trans = []
        wall_pred = []
        #
        combs = []
        for i in range(max_boxes):  #
            rough_info, box, comb = self.autoregressive_decode(boxes, walls=walls, room_mask=room_mask, classes=classes,
            cluster=cluster, combs=combs)  #
            # Check if we have the end symbol
            if box["class_labels"][0, 0, -1] == 1:  #
                break

            #
            check_intersec = is_generate_intersec(box, boxes, bounds)  #
            # check_intersec = False #

            box_label = classes[np.argmax(box["class_labels"].detach().cpu().numpy(), -1).item()]
            if "chair" in box_label:
                check_intersec = False

            for time in range(10):  #
                if check_intersec == False:
                    break
                else:
                    #
                    rough_info, box, comb = self.autoregressive_decode(boxes, walls=walls, room_mask=room_mask,classes=classes,cluster = cluster, combs=combs)
                    check_intersec = is_generate_intersec(box, boxes, bounds)  #

            if box["class_labels"][0, 0, -1] == 1:  #
                break

            #
            if not len(comb) == 0:
                label = classes[np.argmax(box["class_labels"].cpu().numpy(), -1)].item()
                for line in comb["lines"]:
                    if label in line:
                        line[label] -= 1
                        if line[label] == 0:
                            del line[label]

            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)  #
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