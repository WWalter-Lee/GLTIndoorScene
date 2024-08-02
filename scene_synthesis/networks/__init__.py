import torch
try:
    from radam import RAdam
except ImportError:
    pass

from .autoregressive_transformer_rough import AutoregressiveTransformer_Rough
from .autoregressive_transformer_fine import AutoregressiveTransformer_Fine
from .loss import train_on_batch_rough as train_on_batch_simple_autoregressive_rough, \
    validate_on_batch_rough as validate_on_batch_simple_autoregressive_rough,\
    train_on_batch_fine as train_on_batch_simple_autoregressive_fine, \
    validate_on_batch_fine as validate_on_batch_simple_autoregressive_fine

from .hidden_to_fine import AutoregressiveDMLL
from .feature_extractors import get_feature_extractor

def optimizer_factory(config, parameters_rough, parameters_fine):
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = 0.0

    if optimizer == "SGD":
        optimizer_rough = torch.optim.SGD(parameters_rough, lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer_fine = torch.optim.SGD(parameters_fine, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer_rough = torch.optim.Adam(parameters_rough, lr=lr, weight_decay=weight_decay)
        optimizer_fine = torch.optim.Adam(parameters_fine, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamW":
        optimizer_rough = torch.optim.AdamW(parameters_rough, lr=lr, weight_decay=weight_decay)
        optimizer_fine = torch.optim.AdamW(parameters_fine, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        optimizer_rough = torch.optim.RAdam(parameters_rough, lr=lr, weight_decay=weight_decay)
        optimizer_fine = torch.optim.RAdam(parameters_fine, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SparseAdam":
        optimizer_rough = torch.optim.SparseAdam(parameters_rough, lr=lr)
        optimizer_fine = torch.optim.SparseAdam(parameters_fine, lr=lr)
    else:
        raise NotImplementedError()
    return optimizer_rough, optimizer_fine

def build_network(
    input_dims,
    n_classes,
    config,
    weight_file=None,
    device="cpu"):

    network_type = config["network"]["type"]

    if network_type == "autoregressive_transformer":
        train_on_batch_rough = train_on_batch_simple_autoregressive_rough
        validate_on_batch_rough = validate_on_batch_simple_autoregressive_rough
        train_on_batch_fine = train_on_batch_simple_autoregressive_fine
        validate_on_batch_fine = validate_on_batch_simple_autoregressive_fine

        network_rough = AutoregressiveTransformer_Rough(
            input_dims,
            n_classes,
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"],
            config["data"]
        ).to(device)
        network_fine = AutoregressiveTransformer_Fine(
            input_dims,
            n_classes,
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"],
            config["data"]
        ).to(device)
    else:
        raise NotImplementedError()
    if weight_file is not None:
        path, number = weight_file.split("model")
        print("Loading weight file from {}".format(weight_file))
        network_rough.load_state_dict(
            torch.load(path + "model_rough" + number, map_location=device)
        )
        print("rough:", number)
        network_fine.load_state_dict(
            torch.load(path + "model_fine" + number, map_location=device)
        )
        print("fine:", number)

    return network_rough, network_fine, train_on_batch_rough, validate_on_batch_rough, train_on_batch_fine, validate_on_batch_fine
