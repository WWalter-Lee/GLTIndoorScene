#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import argparse
import logging
import os
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader

from training_utils import id_generator, save_experiment_params, load_config
sys.path.append("..")
from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory
from scene_synthesis.stats_logger import StatsLogger, WandB


def yield_forever(iterator):
    while True:
        for x in iterator:
            yield x


def load_checkpoints(model_rough, model_fine, experiment_directory, args, device):
    model_files = [
        f for f in os.listdir(experiment_directory)
        if f.startswith("model_")
    ]
    if len(model_files) == 0:
        return
    ids = [int(f[-5:]) for f in model_files]
    max_id = max(ids)
    model_rough_path = os.path.join(
        experiment_directory, "model_rough_{:05d}"
    ).format(max_id)
    model_fine_path = os.path.join(
        experiment_directory, "model_fine_{:05d}"
    ).format(max_id)

    if not (os.path.exists(model_rough_path)):
        return

    print("Loading rough model checkpoint from {}".format(model_rough_path))
    print("Loading fine model checkpoint from {}".format(model_fine_path))
    model_rough.load_state_dict(torch.load(model_rough_path, map_location=device))
    model_fine.load_state_dict(torch.load(model_fine_path, map_location=device))
    args.continue_from_epoch = max_id+1


def save_checkpoints(epoch, model_rough, model_fine, experiment_directory):
    torch.save(
        model_rough.state_dict(),
        os.path.join(experiment_directory, "model_rough_{:05d}").format(epoch)
    )
    torch.save(
        model_fine.state_dict(),
        os.path.join(experiment_directory, "model_fine_{:05d}").format(epoch)
    )

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="which cuda"
    )
    # parser.add_argument(
    #     "--with_wandb_logger",
    #     action="store_true",
    #     help="Use wandB for logging the training progress"
    # )

    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # # Create an experiment directory using the experiment_tag
    # if args.experiment_tag is None:
    #     experiment_tag = id_generator(9)    # 随机生成9个字符，作为模型输出目录
    # else:
    #     experiment_tag = args.experiment_tag
    #
    # experiment_directory = os.path.join(
    #     args.output_directory,
    #     experiment_tag
    # )
    # 每次debug都会生成新文件夹，先统一在生成在debug文件夹中  # 输出文件夹从args给入
    experiment_directory = args.output_directory  # '../checkpoints/library/debug'

    # if not os.path.exists(experiment_directory):
    #     os.makedirs(experiment_directory)
    #
    # # Save the parameters of this run to a file
    # save_experiment_params(args, experiment_tag, experiment_directory)  # 和git有关，git commit之前会报错，估计需要获取git head
    # print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    train_dataset = get_encoded_dataset(  # 输入3D-FRONT数据，返回AutoregressiveWOCM(序列化数据)
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"])
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    np.savez(
        path_to_bounds,
        sizes=train_dataset.bounds["sizes"],  # {tuple:2}
        translations=train_dataset.bounds["translations"],  # {tuple:2}
        angles=train_dataset.bounds["angles"]  # {tuple:2}(-pi,pi)
    )
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 128),
        num_workers=args.n_processes,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    print("Loaded {} training scenes with {} object types".format(
        len(train_dataset), train_dataset.n_object_types)
    )
    print("Training set has {} bounds".format(train_dataset.bounds))# bounds包含size，translations，angles

    val_loader = DataLoader(
        validation_dataset,
        batch_size=config["validation"].get("batch_size", 1),
        num_workers=args.n_processes,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False
    )
    print("Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )
    print("Validation set has {} bounds".format(validation_dataset.bounds))# 与traindataset一样的bounds

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types


    # Build the network architecture to be used for training
    network_rough, network_fine, train_on_batch_rough, validate_on_batch_rough, train_on_batch_fine, validate_on_batch_fine = build_network(
        train_dataset.feature_size, train_dataset.n_classes,
        config, args.weight_file, device=device
    )

    # Build an optimizer object to compute the gradients of the parameters
    optimizer_rough, optimizer_fine = optimizer_factory(config["training"], network_rough.parameters(), network_fine.parameters())

    # Load the checkpoints if they exist in the experiment directory
    # 加载已有模型继续训练
    # load_checkpoints(network_rough, network_fine, experiment_directory, args, device)

    # # 注释wandb
    # # Initialize the logger
    # if args.with_wandb_logger:
    #     WandB.instance().init(
    #         config,
    #         model_rough=network_rough,
    #         model_fine=network_fine,
    #         project=config["logger"].get(
    #             "project", "autoregressive_transformer"
    #         ),
    #         name="v0.81_diningroom",
    #         watch=False,
    #         log_frequency=10
    #     )

    # Log the stats to a file
    # StatsLogger.instance().add_output_file(open(
    #     os.path.join(experiment_directory, "log.txt"),
    #     "w"
    # ))

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)
    save_every=5

    # Do the training
    for i in range(args.continue_from_epoch, epochs):
        network_rough.train()
        network_fine.train()
        # network_fine.eval()
        for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
            # Move everything to device
            for k, v in sample.items():
                sample[k] = v.to(device)
            rough_loss = train_on_batch_rough(network_rough, optimizer_rough, sample, config)  # sample为一个批次的数据+gt
            fine_loss = train_on_batch_fine(network_fine, optimizer_fine, sample, config)

            batch_loss = rough_loss + fine_loss
            # batch_loss = rough_loss
            StatsLogger.instance().print_progress(i+1, b+1, batch_loss)

        if (i % save_every) == 0:
            save_checkpoints(
                i,
                network_rough,
                network_fine,
                experiment_directory,
            )
        StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            print("====> Validation Epoch ====>")
            network_rough.eval()
            network_fine.eval()
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    sample[k] = v.to(device)
                rough_loss = validate_on_batch_rough(network_rough, sample, config)
                fine_loss = validate_on_batch_fine(network_fine,sample,config)
                batch_loss = rough_loss + fine_loss
                # batch_loss=rough_loss
                StatsLogger.instance().print_progress(-1, b+1, batch_loss)
            StatsLogger.instance().clear()
            print("====> Validation Epoch ====>")

if __name__ == "__main__":
    main(sys.argv[1:])
