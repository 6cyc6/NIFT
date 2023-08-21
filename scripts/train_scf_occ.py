import sys
import os, os.path as osp
import configargparse
import torch
from torch.utils.data import DataLoader


import src.model.vnn_occupancy_scf_net as vnn_occupancy_scf_net
from src.training import summaries, losses, training, dataio

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logging', help='root for logging')
p.add_argument('--obj_class', type=str, required=True,
               help='bottle, mug, bowl, all')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--sidelength', type=int, default=128)

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=40001,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=10000,
               help='Training steps until save checkpoint')

p.add_argument('--depth_aug', action='store_true', help='depth_augmentation')
p.add_argument('--multiview_aug', action='store_true', help='multiview_augmentation')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--dgcnn', action='store_true', help='If you want to use a DGCNN encoder instead of pointnet (requires more GPU memory)')

p.add_argument('--o_dim', type=int, default=5)
opt = p.parse_args()

train_dataset = dataio.JointOccScfTrainDataset(opt.sidelength, o_dim=opt.o_dim, obj_class=opt.obj_class)
val_dataset = dataio.JointOccScfTrainDataset(opt.sidelength, o_dim=opt.o_dim, phase='val', obj_class=opt.obj_class)


train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              drop_last=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                            drop_last=True, num_workers=4)

model = vnn_occupancy_scf_net.VNNOccScfNet(latent_dim=256, o_dim=opt.o_dim, sigmoid=True).cuda()

if opt.checkpoint_path is not None:
    model.load_state_dict(torch.load(opt.checkpoint_path))

# model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model_parallel = model
print(model)

# Define the loss
root_path = os.path.join(opt.logging_root, opt.experiment_name)

# Define the loss
summary_fn = summaries.scf_net
root_path = os.path.join(opt.logging_root, opt.experiment_name)
loss_fn = val_loss_fn = losses.scf_net

training.train_scf_occ(model=model_parallel, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs,
                       lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                       model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                       clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True)

