

import torch
import torch.nn as nn
import numpy as np

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import LengthMask

from .base import FixedPositionalEncoding
torch.autograd.set_detect_anomaly(True)
from .hidden_to_rough import AutoregressiveDMLL_Rough
from .bbox_output import AutoregressiveOutput_Rough

class AutoregressiveTransformer_Rough(nn.Module):
    def __init__(self, input_dims, n_classes, feature_extractor, config, config_data):
        super().__init__()
        self.transformer_encoder_rough = TransformerEncoderBuilder.from_kwargs(
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 12),
            query_dimensions=config.get("query_dimensions", 64),
            value_dimensions=config.get("value_dimensions", 64),
            feed_forward_dimensions=config.get(
                "feed_forward_dimensions", 3072
            ),
            attention_type="full",
            activation="gelu"
        ).get()
        hidden_dims = config.get("hidden_dims", 768)

        self.feature_extractor = feature_extractor
        self.fc_room_f = nn.Linear(
            self.feature_extractor.feature_size, 64 * 7
        )
        self.input_dims = input_dims  #
        self.n_classes = self.input_dims - 3 - 3 - 1  #
        self.fc_class = nn.Linear(self.n_classes, 64, bias=False)
        #
        self.pe_pos = FixedPositionalEncoding(proj_dims=64)  # x, y, z
        self.pe_size = FixedPositionalEncoding(proj_dims=64)  # x,y,z
        self.pe_angle_z = FixedPositionalEncoding(proj_dims=64)  # z

        self.fc = nn.Linear(64 * 7, hidden_dims)

        self.hidden2rough = AutoregressiveDMLL_Rough(  # DMLL, Distributed Maximum Log-Likelihood
            config.get("hidden_dims", 768),  #
            n_classes,
            config.get("n_mixtures", 4),  # 10
            AutoregressiveOutput_Rough,
            config.get("with_extra_fc", False))

        self.register_parameter(
            "empty_token_embedding_rough", nn.Parameter(torch.randn(1, 64*7))
        )

    def room_mask_features(self, room_mask):
        room_layout_f = self.fc_room_f(self.feature_extractor(room_mask))
        return room_layout_f[:, None, :]

    def forward(self, sample_params):
        # Unpack the sample_params
        class_labels = sample_params["class_labels"]
        sizes = sample_params["sizes"]
        trans = sample_params["translations"]
        room_layout = sample_params["room_layout"]
        B, _, _ = class_labels.shape

        # room mask
        room_mask_f = self.room_mask_features(room_layout)

        class_f = self.fc_class(class_labels)

        trans_x = self.pe_pos(trans[:, :, 0:1])
        trans_y = self.pe_pos(trans[:, :, 1:2])
        trans_z = self.pe_pos(trans[:, :, 2:3])
        trans_f = torch.concat((trans_x, trans_y, trans_z), dim=2)

        size_f_x = self.pe_size(sizes[:, :, 0:1])
        size_f_y = self.pe_size(sizes[:, :, 1:2])
        size_f_z = self.pe_size(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        X = torch.cat([class_f, trans_f, size_f], dim=-1)

        X = torch.cat([room_mask_f, self.empty_token_embedding_rough.expand(B, -1, -1), X], dim=1)
        X = self.fc(X)

        lengths_rough = LengthMask(
            sample_params["lengths"] + 2,  # +2:room_mask,q
            max_len=X.shape[1]  # 14
        )
        F = self.transformer_encoder_rough(X, length_mask=lengths_rough)
        q_rough = F[:, 1:2]  # q^
        rough_pred = self.hidden2rough(q_rough, sample_params)

        return rough_pred

    def _encode_rough(self, boxes, walls, room_mask):
        class_labels = boxes["class_labels"][:,1:]
        translations = boxes["translations"][:,1:]  #
        sizes = boxes["sizes"][:, 1:]
        B, _, _ = class_labels.shape

        walls_num = len(walls["points"])

        # room mask token
        room_mask_f = self.room_mask_features(room_mask)
        # check_roommask=room_mask.cpu().numpy() #

        # seq tokens
        class_f = self.fc_class(class_labels)
        trans_x = self.pe_pos(translations[:, :, 0:1])
        trans_y = self.pe_pos(translations[:, :, 1:2])
        trans_z = self.pe_pos(translations[:, :, 2:3])
        trans_f = torch.concat((trans_x, trans_y, trans_z), dim=2)

        size_f_x = self.pe_size(sizes[:, :, 0:1])
        size_f_y = self.pe_size(sizes[:, :, 1:2])
        size_f_z = self.pe_size(sizes[:, :, 2:3])
        size_f = torch.cat([size_f_x, size_f_y, size_f_z], dim=-1)

        X = torch.cat([class_f, trans_f, size_f], dim=-1)
        X = torch.cat([room_mask_f, self.empty_token_embedding_rough.expand(B, -1, -1), X], dim=1)
        X = self.fc(X)  # ->512

        F = self.transformer_encoder_rough(X, length_mask=None)  # (1,2,512)->(1,2,512)
        q_rough = F[:, 1:2]
        #
        rough_class = self.hidden2rough.sample_class_labels(q_rough)
        rough_trans = self.hidden2rough.sample_translations(q_rough, rough_class)
        rough_size = self.hidden2rough.sample_sizes(q_rough, rough_class, rough_trans)
        #
        rough_info = {"rough_class": rough_class, "rough_trans": rough_trans, "rough_size": rough_size}
        #
        wall_points = np.sum(walls["points"], axis=1) / 2
        rough_points = rough_trans.cpu().numpy().reshape(1, 3).repeat(walls_num, axis=0)
        dis = np.sum((rough_points - wall_points) ** 2, axis=1)
        wall_pred = torch.tensor([dis.argmin() + 1]).reshape(1,1).to(q_rough.device) #
        rough_info["wall_pred"] = wall_pred
        return rough_info

    @torch.no_grad()
    def distribution_classes(self, boxes, room_mask, device="cpu"):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())
        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Add the start box token in the beginning
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)
        return self.hidden2output.pred_class_probs(F)

    @torch.no_grad()
    def distribution_translations(
        self,
        boxes,
        room_mask, 
        class_label,
        device="cpu"
    ):
        # Shallow copy the input dictionary
        boxes = dict(boxes.items())

        # Make sure that the provided class_label will have the correct format
        if isinstance(class_label, int):
            one_hot = torch.eye(self.n_classes)
            class_label = one_hot[class_label][None, None]
        elif not torch.is_tensor(class_label):
            class_label = torch.from_numpy(class_label)

        # Make sure that the class label the correct size,
        # namely (batch_size, 1, n_classes)
        assert class_label.shape == (1, 1, self.n_classes)

        # Create the initial input to the transformer, namely the start token
        start_box = self.start_symbol(device)
        # Concatenate to the given input (that's why we shallow copy in the
        # beginning of this method
        for k in start_box.keys():
            boxes[k] = torch.cat([start_box[k], boxes[k]], dim=1)

        # Compute the features using the transformer
        F = self._encode(boxes, room_mask)

        # Get the dmll params for the translations
        return self.hidden2output.pred_dmll_params_translation(
            F, class_label
        )
