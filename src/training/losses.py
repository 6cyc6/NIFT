import torch
import torch.nn.functional as F


def occ_scf_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()

    # occupancy loss BCE
    label_occ = ground_truth['occ'].squeeze()
    label_occ = (label_occ + 1) / 2.

    loss_dict['occ'] = -1 * (label_occ * torch.log(model_outputs['occ'] + 1e-5) + (1 - label_occ) * torch.log(
        1 - model_outputs['occ'] + 1e-5)).mean()

    # scf loss L1
    label_scf = ground_truth['scf'].squeeze()

    loss_dict['scf'] = F.l1_loss(model_outputs['scf'], label_scf)

    return loss_dict


def sdf_scf_net(model_outputs, ground_truth, val=False, delta=0.3, scaling=10.0):
    loss_dict = dict()

    # sdf loss clamped L1
    label_sdf = ground_truth['sdf'].squeeze()
    clamped_label = torch.clip(label_sdf * scaling, -delta * scaling, delta * scaling)
    output_sdf = model_outputs["sdf"]
    clamped_output = torch.clip(output_sdf * scaling, -delta * scaling, delta * scaling)

    loss_dict['sdf'] = torch.abs(clamped_label - clamped_output).mean()

    # scf loss L1
    label_scf = ground_truth['scf'].squeeze()

    loss_dict['scf'] = F.l1_loss(model_outputs['scf'], label_scf)

    return loss_dict


def occ_sdf_scf_net(model_outputs, ground_truth, val=False, delta=0.3, scaling=10.0):
    loss_dict = dict()

    # occupancy loss BCE
    label_occ = ground_truth['occ'].squeeze()
    label_occ = (label_occ + 1) / 2.

    loss_dict['occ'] = -1 * (label_occ * torch.log(model_outputs['occ'] + 1e-5) + (1 - label_occ) * torch.log(
        1 - model_outputs['occ'] + 1e-5)).mean()

    # sdf loss clamped L1
    label_sdf = ground_truth['sdf'].squeeze()
    clamped_label = torch.clip(label_sdf * scaling, -delta * scaling, delta * scaling)
    output_sdf = model_outputs["sdf"]
    clamped_output = torch.clip(output_sdf * scaling, -delta * scaling, delta * scaling)

    loss_dict['sdf'] = torch.abs(clamped_label - clamped_output).mean()

    # scf loss L1
    label_scf = ground_truth['scf'].squeeze()
    loss_dict['scf'] = F.l1_loss(model_outputs['scf'], label_scf)

    return loss_dict


def scf_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    # dist = F.l1_loss(model_outputs['occ'],label)
    loss_dict['scf'] = F.l1_loss(model_outputs['occ'], label)

    return loss_dict


def occupancy(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def occupancy_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def distance_net(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ'].squeeze()

    dist = torch.abs(model_outputs['occ'] - label * 100).mean()
    loss_dict['dist'] = dist

    return loss_dict


def semantic(model_outputs, ground_truth, val=False):
    loss_dict = {}

    label = ground_truth['occ']
    label = ((label + 1) / 2.).squeeze()

    if val:
        loss_dict['occ'] = torch.zeros(1)
    else:
        loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'].squeeze() + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'].squeeze() + 1e-5)).mean()

    return loss_dict


# source activate lndf
# cd project/lndf_robot/src/ndf_robot
#
# export PYTHONPATH=$PYTHONPATH:`pwd`
# export LNDF_SOURCE_DIR=/home/kit/anthropomatik/qy7043/project/ndf_robot/src/ndf_robot/data
# export LNDF_WEIGHT=$PWD/model_weights
#
# python training/train_conv_occupancy_net_nift.py --obj_class mug --experiment_name test_lndf_nift --num_epochs 50 \
# --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 16 --triplet_loss --multiview_aug --checkpoint_path lnift_test.pth