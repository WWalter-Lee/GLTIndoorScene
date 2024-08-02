import torch
import torch.nn as nn
import numpy as np

from .Encoder import TransformerEncoder

from .base import FixedPositionalEncoding
torch.autograd.set_detect_anomaly(True)
from .hidden_to_fine import AutoregressiveDMLL
from .bbox_output import AutoregressiveBBoxOutput
from ..datasets.global_ref_transform import global2ref

class AutoregressiveTransformer_Fine(nn.Module):
    def __init__(self, input_dims, n_classes, feature_extractor, config, config_data):
        super().__init__()
        self.config_data = config_data
        hidden_dims = config.get("hidden_dims", 768)
        self.transformer_encoder_fine = TransformerEncoder(
            d_model=config.get(
                "feed_forward_dimensions", 3072
            ),
            heads=config.get("n_heads", 12),
            dropout=0.1,
            activation=nn.GELU(),
            layer_num=config.get("n_layers", 6)
        )

        self.walls_max = int(self.config_data["walls_max"]) + 1

        self.pe_pos = FixedPositionalEncoding(proj_dims=64)  # x, y, z
        self.pe_size = FixedPositionalEncoding(proj_dims=64)  # x, y, z
        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)  # z
        self.fc_wall_token_length = FixedPositionalEncoding(proj_dims=64)

        self.input_dims = input_dims
        self.n_classes = self.input_dims - 3 - 3 - 1
        self.fc_class = nn.Linear(self.n_classes, 64, bias=False)

        self.fc_walls_ref = nn.Linear(64 * 1, hidden_dims)

        self.register_parameter(
            "empty_token_embedding_fine", nn.Parameter(torch.randn(1, 512))
        )

        self.fc_rough_token = nn.Linear(64 * 4, hidden_dims)
        self.fc_ref = nn.Linear(512, hidden_dims)

        self.hidden2fine = AutoregressiveDMLL(  # DMLL, Distributed Maximum Log-Likelihood
            config.get("hidden_dims", 768),
            n_classes,
            config.get("n_mixtures", 4),
            AutoregressiveBBoxOutput,
            config.get("with_extra_fc", False))
    def forward(self, sample_params):

        B, _, _ = sample_params["class_labels"].shape

        wall_tr = sample_params["walls_order_tr"].to(int)
        wall_idx = (torch.arange(B), wall_tr.squeeze(dim=1))
        sample_params["ref_class_labels"] = sample_params["ref_class_labels"][wall_idx]
        sample_params["ref_translations"] = sample_params["ref_translations"][wall_idx]
        sample_params["ref_sizes"] = sample_params["ref_sizes"][wall_idx]
        sample_params["ref_angles"] = sample_params["ref_angles"][wall_idx]
        sample_params["ref_class_labels_tr"] = sample_params["ref_class_labels_tr"][wall_idx]
        sample_params["ref_translations_tr"] = sample_params["ref_translations_tr"][wall_idx]
        sample_params["ref_sizes_tr"] = sample_params["ref_sizes_tr"][wall_idx]
        sample_params["ref_angles_tr"] = sample_params["ref_angles_tr"][wall_idx]

        # seq token
        ref_translations = sample_params["ref_translations"]
        ref_sizes = sample_params["ref_sizes"]
        ref_angles = sample_params["ref_angles"]
        ref_class = sample_params["ref_class_labels"]

        class_f = self.fc_class(ref_class)

        pos_f_x = self.pe_pos(ref_translations[:, :, 0:1])
        pos_f_y = self.pe_pos(ref_translations[:, :, 1:2])
        pos_f_z = self.pe_pos(ref_translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        size_f_x = self.pe_size(ref_sizes[:, :, 0:1])
        size_f_y = self.pe_size(ref_sizes[:, :, 1:2])
        size_f_z = self.pe_size(ref_sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        angle_f = self.pe_angle_z(ref_angles)
        X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)

        # wall token
        wall_length_ref = sample_params["wall_length_ref"][:, :, None]
        walls_length = self.fc_wall_token_length(wall_length_ref)

        wall_length_ref = walls_length[wall_idx]
        walls_feature = wall_length_ref[:, None, :]
        walls_token_ref = self.fc_walls_ref(walls_feature)

        # rough token
        box_class_tr = sample_params["class_labels_tr"]
        box_size_tr = sample_params["sizes_tr"]

        box_class = self.fc_class(box_class_tr)
        size_x = self.pe_size(box_size_tr[:, :, 0:1])
        size_y = self.pe_pos(box_size_tr[:, :, 1:2])
        size_z = self.pe_pos(box_size_tr[:, :, 2:3])
        size = torch.concat((size_x, size_y, size_z), dim=2)

        rough_token = self.fc_rough_token(torch.concat((box_class, size), dim=2))

        X = torch.cat([walls_token_ref, rough_token, self.empty_token_embedding_fine.expand(B, -1, -1), X],dim=1)
        X = self.fc_ref(X)

        mask = torch.arange(X.shape[1], device=X.device).reshape(1, -1).repeat(B, 1)
        length = sample_params["lengths"].reshape(-1, 1) + 3  # +3: wallstoken, roughtoken, q
        global_mask = (mask >= length).int()

        walls_order = sample_params["walls_order"]
        ref_mask = walls_order == wall_tr
        ref_mask_zero = torch.ones((ref_mask.shape[0], 3), device=ref_mask.device)
        ref_mask = torch.cat((ref_mask_zero, ref_mask), dim=1).bool()
        ref_mask = ~ ref_mask

        ref_mask = (global_mask + ref_mask).bool()
        ref_mask = ~ ref_mask
        ref_mask = ref_mask[:, :, None] * ref_mask[:, None, :]
        ref_mask = ~ ref_mask

        F = self.transformer_encoder_fine(X, ref_mask=ref_mask)
        q_ref = F[:, 2:3]
        ref_pred = self.hidden2fine(q_ref, sample_params)  # AutoregressiveDMLL

        return ref_pred

    def _encode_fine(self, boxes, walls, rough_info):
        class_labels = boxes["class_labels"][:, 1:]
        translations = boxes["translations"][:, 1:]
        sizes = boxes["sizes"][:, 1:]
        angles = boxes["angles"][:, 1:]
        walls_order = boxes["walls_order"][:, 1:]
        B, _, _ = class_labels.shape

        walls_num = len(walls["points"])
        walls_max = self.walls_max - 1

        if(walls_max - walls_num) < 0:
            print(1)
        wall_points = np.vstack((np.zeros([1, 3]), walls["origin_scaled"], np.zeros((walls_max - walls_num, 3))))
        wall_points = torch.from_numpy(wall_points).float().to(sizes.device)
        wall_points = wall_points[None,:,:]

        # wall_token
        wall_pred = rough_info["wall_pred"]
        walls_token_length = np.concatenate((-np.ones(1), walls["length"], -np.ones(walls_max - walls_num)))
        walls_token_length = torch.from_numpy(walls_token_length).float().to(sizes.device)[None, :, None]  # (1,9,1)
        walls_length = self.fc_wall_token_length(walls_token_length)
        wall_length_ref = walls_length[0][wall_pred]

        walls_feature = wall_length_ref
        walls_token_ref = self.fc_walls_ref(walls_feature)


        wall_basis = walls["basis"][wall_pred - 1]  #
        wall_basis_T = wall_basis.T  #
        origin_scaled = walls["origin_scaled"][wall_pred-1]

        # rough token
        rough_class = rough_info["rough_class"]
        rough_size = rough_info["rough_size"]
        box_class = self.fc_class(rough_class)

        size_x = self.pe_size(rough_size[:, :, 0:1])
        size_y = self.pe_pos(rough_size[:, :, 1:2])
        size_z = self.pe_pos(rough_size[:, :, 2:3])
        size = torch.concat((size_x, size_y, size_z),dim=2)

        rough_token = self.fc_rough_token(torch.concat((box_class, size), dim=2))

        # seq token
        # class_f, size_f
        class_f = self.fc_class(class_labels)
        size_f_x = self.pe_size(sizes[:, :, 0:1])
        size_f_y = self.pe_size(sizes[:, :, 1:2])
        size_f_z = self.pe_size(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        #
        translations = translations.cpu().numpy().reshape(-1,3)
        ref_translations = global2ref(wall_basis_T, origin_scaled, translations)
        ref_translations = torch.from_numpy(ref_translations).float().to(sizes.device)[None, :, :]
        pos_f_x = self.pe_pos(ref_translations[:, :, 0:1])
        pos_f_y = self.pe_pos(ref_translations[:, :, 1:2])
        pos_f_z = self.pe_pos(ref_translations[:, :, 2:3])
        pos_f = torch.cat([pos_f_x, pos_f_y, pos_f_z], dim=-1)

        #
        wall_basis_x = wall_basis[0]
        x = np.array([1, 0, 0])
        cos_theta = np.dot(wall_basis_x, x)  #
        theta = np.arccos(cos_theta)
        #
        ore = np.cross(wall_basis_x, x)  #
        if ore[1] < 0:
            theta = -theta
        angle_min = -np.pi
        angles = angles.cpu().numpy().reshape(-1,1)
        ref_angles = (angles + theta - angle_min) % (2 * np.pi) + angle_min
        ref_angles = torch.from_numpy(ref_angles).float().to(sizes.device)[None, :, :]

        angle_f = self.pe_angle_z(ref_angles)

        X = torch.cat([class_f, pos_f, size_f, angle_f], dim=-1)
        X = torch.cat([walls_token_ref, rough_token, self.empty_token_embedding_fine.expand(B, -1, -1), X], dim=1)
        X = self.fc_ref(X)

        # global_mask
        mask = torch.arange(X.shape[1], device=X.device).reshape(1, -1).repeat(B, 1)
        length = angles.shape[1] + 3  # +3: wallstoken, roughtoken,q
        global_mask = (mask >= length).int()

        # #
        walls_order = torch.argmax(walls_order, dim=2) #
        # #
        ref_mask = (walls_order == wall_pred)  #
        ref_mask_zero = torch.ones((ref_mask.shape[0], 3), device=ref_mask.device)  #
        ref_mask = torch.cat((ref_mask_zero, ref_mask), dim=1).bool()
        ref_mask = ~ ref_mask

        ref_mask = (global_mask + ref_mask).bool()
        ref_mask = ~ ref_mask  #
        ref_mask = ref_mask[:,:,None] * ref_mask[:,None,:] # (1,21,1)*(1,1,21)=(1,21,21)
        ref_mask = ~ ref_mask  #

        F = self.transformer_encoder_fine(X, ref_mask=ref_mask)[:, 2:3]  #

        return (wall_basis, origin_scaled, F)

