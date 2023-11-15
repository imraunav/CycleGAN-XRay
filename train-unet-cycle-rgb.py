import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
import numpy as np
from matplotlib import pyplot as plt

from models.unet import UNet

# from models.unet_sp import UNet_SP
# from models.conv_discriminator import Discriminator
from utils.dataset import XRayDataset

import hyperparameters


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # any unused port

    # initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_loader(world_size):
    dataset = XRayDataset(
        hyperparameters.dataset_path
    )  # define the class and then use function
    sampler = DistributedSampler(dataset, num_replicas=world_size, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=hyperparameters.num_workers,
        sampler=sampler,
    )
    return loader, sampler


def main(rank, world_size):
    print(f"Running training on GPU {rank}")
    ddp_setup(rank, world_size)

    g23 = UNet(n_channels=2, n_classes=3).to(rank)
    g23 = DDP(g23, device_ids=[rank])

    g32 = UNet(n_channels=3, n_classes=2).to(rank)
    g32 = DDP(g32, device_ids=[rank])

    dataloader, datasampler = get_loader(world_size)
    trainer = Trainer(rank, g23, g32, dataloader, datasampler)
    trainer.train(hyperparameters.max_epochs)
    cleanup()


class Trainer:
    def __init__(
        self,
        gpu_id,
        g23,
        g32,
        dataloader,
        datasampler,
    ) -> None:
        print(f"Initializing trainer on GPU {gpu_id}")
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.g23 = g23
        self.g32 = g32
        self.dataloader = dataloader
        self.datasampler = datasampler

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.optimizer = ZeroRedundancyOptimizer(
            list(self.g32.parameters()) + list(self.g23.parameters()),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )
        self.lrschedule = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            threshold=0.001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
        )

    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
    ):
        print(f"Checkpoint reached at epoch {epoch}!")

        if not os.path.exists("./weights"):
            os.mkdir("./weights")

        ckp = self.g23.module.state_dict()
        model_path = f"./weights/g23unetrgb_{epoch}_loss{loss:.4f}.pt"
        torch.save(ckp, model_path)

        ckp = self.g32.module.state_dict()
        model_path = f"./weights/g32unetrgb_{epoch}_loss{loss:.4f}.pt"
        torch.save(ckp, model_path)

    def _on_epoch(self, epoch: int):
        self.datasampler.set_epoch(epoch)
        epoch_loss = []
        for batch in self.dataloader:
            batch_loss = self._on_batch(batch)
            epoch_loss.append(batch_loss)

        return np.mean(epoch_loss)

    def _on_batch(self, batch):
        low_imgs, high_imgs = batch
        low_imgs = low_imgs.to(self.gpu_id)
        high_imgs = high_imgs.to(self.gpu_id)
        losses = []
        # for imgs in [low_imgs, high_imgs]:
        imgs = torch.concat([low_imgs, high_imgs], dim=-3)
        fused_imgs = self.g23(imgs).sigmoid()
        back_imgs = self.g32(fused_imgs).sigmoid()


        # plt.subplot(3,2,1)
        # plt.imshow(low_imgs[0, 1].numpy())
        # plt.subplot(3,2,2)
        # plt.imshow(high_imgs[0, 1].numpy())
        # plt.subplot(3,2,3)
        # plt.imshow(fused_imgs[0, 1].numpy())
        # plt.subplot(3,2,5)
        # plt.imshow(back_imgs[0, 1].numpy())
        # plt.subplot(3,2,6)
        # plt.imshow(back_imgs[0, 2].numpy())
        # plt.savefig("UNet-sample.png")
        
        loss = self.mse_loss(imgs, back_imgs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())

        return np.mean(losses)

    def train(self, max_epoch):
        self.g23.train()
        self.g32.train()

        for epoch in range(max_epoch):
            epoch_loss = self._on_epoch(epoch)
            self.lrschedule.step(epoch_loss)
            self.loss_writer(epoch, epoch_loss)
            if epoch % hyperparameters.ckpt_per == 0 and self.gpu_id == 0:
                self._save_checkpoint(epoch, epoch_loss)

        # Final epoch save
        if self.gpu_id == 0:
            self._save_checkpoint(max_epoch, epoch_loss)

    def loss_writer(self, epoch, epoch_loss):
        print(f"[GPU:{self.gpu_id}] - Epoch:{epoch} - Loss:{epoch_loss}")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )  #
