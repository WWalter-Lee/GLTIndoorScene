import torch
from .base import FixedPositionalEncoding, sample_from_dmll
from ..datasets.global_ref_transform import global2ref,ref2global_tensor
from ..datasets.utils import scale, descale
import os,json
from ..stats_logger import StatsLogger

def train_on_batch_rough(model_rough, optimizer_rough, sample_params, config):
    # Make sure that everything has the correct size
    optimizer_rough.zero_grad()

    X_rough_pred = model_rough(sample_params)  #

    loss_rough = X_rough_pred.reconstruction_loss_rough(sample_params)
    # Do the backpropagation
    loss_rough.backward()
    # Do the update
    optimizer_rough.step()

    return loss_rough.item()


@torch.no_grad()
def validate_on_batch_rough(model_rough, sample_params, config):
    X_rough_pred = model_rough(sample_params)  #
    loss_rough = X_rough_pred.reconstruction_loss_rough(sample_params)
    return loss_rough.item()

def train_on_batch_fine(model_fine, optimizer_fine, sample_params, config):
    # Make sure that everything has the correct size
    optimizer_fine.zero_grad()

    X_fine_pred = model_fine(sample_params)  #
    loss_fine = X_fine_pred.reconstruction_loss_fine(sample_params)
    #
    # loss_ergo = compute_ergoloss(X_fine_pred,sample_params,config)
    # loss_ergo *= 100

    # StatsLogger.instance()["loss_ergo"].value = loss_ergo.item()
    # loss_total = loss_fine + loss_ergo
    # # Do the backpropagation
    # loss_total.backward()  #
    loss_fine.backward()  #

    # # Do the update
    optimizer_fine.step()

    # return loss_fine.item(), loss_ergo.item()
    return loss_fine.item()


@torch.no_grad()
def validate_on_batch_fine(model_fine, sample_params, config):
    X_fine_pred = model_fine(sample_params)  #
    loss_fine = X_fine_pred.reconstruction_loss_fine(sample_params)
    #
    # loss_ergo = compute_ergoloss(X_fine_pred, sample_params, config)
    # StatsLogger.instance()["loss_ergo"].value = loss_ergo.item()

    # return loss_fine.item(), loss_ergo.item()
    return loss_fine.item()

def ref2global(sample_params,tran_ref,i,bounds):
    #
    wall_idx = int(sample_params["walls_order_tr"][i].item())
    wall_basis = sample_params["walls_basis"][i][wall_idx]
    origin_scaled = sample_params["walls_origin_scaled"][i][wall_idx]
    tran = ref2global_tensor(wall_basis.T, origin_scaled, tran_ref[None, :])  #
    tran = descale(tran, bounds[0], bounds[1])  #

    return tran
