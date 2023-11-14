import os
import pickle
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
import numpy as np
from matplotlib import pyplot as plt

from models.unet import UNet
from models.discriminator import UNet_sn

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

    g21 = UNet(n_channels=2, n_classes=1).to(rank)
    g21.load_state_dict(torch.load(hyperparameters.pretrain_weights21))
    g21 = DDP(g21, device_ids=[rank])

    g12 = UNet(n_channels=1, n_classes=2).to(rank)
    g12.load_state_dict(torch.load(hyperparameters.pretrain_weights12))
    g12 = DDP(g12, device_ids=[rank])

    d = UNet_sn(1, 1).to(rank)
    d = DDP(d, device_ids=[rank])

    dataloader, datasampler = get_loader(world_size)
    trainer = Trainer(rank, g21, g12, d, dataloader, datasampler)
    trainer.train(hyperparameters.max_epochs)
    cleanup()


class Trainer:
    def __init__(self, gpu_id, g21, g12, d, dataloader, datasampler) -> None:
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        torch.cuda.set_device(gpu_id)  # master gpu takes up extra memory
        torch.cuda.empty_cache()

        self.gpu_id = gpu_id
        self.g21 = g21
        self.g12 = g12
        self.d = d
        self.dataloader = dataloader
        self.datasampler = datasampler

        self.adv_crit = nn.BCELoss()
        self.cycle_crit = nn.L1Loss()

        self.cycle_optimizer = ZeroRedundancyOptimizer(
            list(self.g12.parameters()) + list(self.g21.parameters()),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )

        self.disc_optimizer = ZeroRedundancyOptimizer(
            self.d.parameters(),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )

        self.gen_optimizer = ZeroRedundancyOptimizer(
            self.g21.parameters(),
            optimizer_class=optim.Adam,
            lr=hyperparameters.base_learning_rate,
        )

    def _save_checkpoint(
        self,
        epoch: int,
    ):
        print(f"Checkpoint reached at epoch {epoch}!")

        if not os.path.exists("./weights"):
            os.mkdir("./weights")

        ckp = self.g21.module.state_dict()
        model_path = f"./weights/g21unetgan_{epoch}.pt"
        torch.save(ckp, model_path)

        ckp = self.g12.module.state_dict()
        model_path = f"./weights/g12unetgan_{epoch}.pt"
        torch.save(ckp, model_path)

        ckp = self.d.module.state_dict()
        model_path = f"./weights/dunetgan_{epoch}.pt"
        torch.save(ckp, model_path)

    def loss_writer(self, epoch, epoch_loss):
        print(f"[GPU:{self.gpu_id}] - Epoch:{epoch} - Loss:{epoch_loss}")

    def update_discriminator(self, real_batch, fake_batch):
        self.disc_optimizer.zero_grad()
        # predictions
        real_pred = torch.sigmoid(self.d(real_batch))
        fake_pred = torch.sigmoid(self.d(fake_batch.detach()))

        # prep labels
        real_labels = torch.full(
            real_pred.shape,
            1,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )
        fake_labels = torch.full(
            fake_pred.shape,
            0,
            dtype=torch.float32,
            device=self.gpu_id,
            requires_grad=False,
        )

        real_loss = self.adv_crit(real_pred, real_labels)
        fake_loss = self.adv_crit(fake_pred, fake_labels)
        loss = real_loss + fake_loss
        loss.backward()
        self.disc_optimizer.step()
        return loss.item()

    def update_generator(self, imgs, real_batch):
        self.gen_optimizer.zero_grad()
        # generate
        fake_batch = self.g21(imgs)

        # classify
        pred = self.d(fake_batch)
        target = torch.full(
            pred.shape, 1, dtype=torch.float32, device=self.gpu_id, requires_grad=False
        )
        loss = self.adv_crit(pred, target)
        loss.backward()
        self.gen_optimizer.step()
        return loss.item()

    def update_cyclic(self, imgs):
        self.cycle_optimizer.zero_grad()
        fused = torch.sigmoid(self.g21(imgs))
        back = torch.sigmoid(self.g12(fused))

        loss = self.cycle_crit(fused, back)
        loss.backward()
        self.cycle_optimizer.step()

        return loss.item()

    def _on_batch(self, batch):
        low_imgs, high_imgs = batch
        low_imgs = low_imgs.to(self.gpu_id)
        high_imgs = high_imgs.to(self.gpu_id)

        imgs = torch.concat([low_imgs, high_imgs], dim=-3)
        # update discriminator
        fused_imgs = self.g21(imgs)  # using BCElogits
        if torch.randn(1).item() < 0.5:  # randomly choose which the real images are
            d_loss = self.update_discriminator(low_imgs, fused_imgs)
        else:
            d_loss = self.update_discriminator(high_imgs, fused_imgs)

        # update generator
        g_loss = self.update_generator(imgs)

        # update cyclic loss
        cyc_loss = self.update_cyclic(imgs)

        return d_loss, g_loss, cyc_loss

    def _on_epoch(self, epoch):
        self.datasampler.set_epoch(epoch)
        d_losses, g_losses, cyc_losses = [], [], []
        for batch in self.dataloader:
            with torch.autograd.set_detect_anomaly(True):
                d_loss, g_loss, cyc_loss = self._on_batch(batch)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            cyc_losses.append(cyc_loss)

        return d_losses, g_losses, cyc_losses

    def train(self, max_epoch):
        self.g21.train()
        self.g12.train()
        self.d.train()

        d_losses, g_losses, cyc_losses = [], [], []
        for epoch in range(max_epoch):
            epoch_d_losses, epoch_g_losses, epoch_cyc_losses = self._on_epoch(epoch)
            d_losses.extend(epoch_d_losses)
            g_losses.extend(epoch_g_losses)
            cyc_losses.extend(epoch_cyc_losses)
            if epoch < 50:
                print(
                    f"[GPU{self.gpu_id}] Epoch:{epoch} d_loss:{d_losses[-1]}, g_loss:{g_losses[-1]}, cyc_loss:{cyc_losses[-1]}"
                )
            if epoch % hyperparameters.ckpt_per == 0 and self.gpu_id == 0:
                with open("d_loss.pkl", mode="wb") as file:
                    pickle.dump(d_losses, file, pickle.HIGHEST_PROTOCOL)
                with open("g_loss.pkl", mode="wb") as file:
                    pickle.dump(g_losses, file, pickle.HIGHEST_PROTOCOL)
                with open("cyc_loss.pkl", mode="wb") as file:
                    pickle.dump(cyc_losses, file, pickle.HIGHEST_PROTOCOL)
                self._save_checkpoint(epoch)

        # Final epoch save
        if self.gpu_id == 0:
            with open("d_loss.pkl", mode="wb") as file:
                pickle.dump(d_losses, file, pickle.HIGHEST_PROTOCOL)
            with open("g_loss.pkl", mode="wb") as file:
                pickle.dump(g_losses, file, pickle.HIGHEST_PROTOCOL)
            with open("cyc_loss.pkl", mode="wb") as file:
                pickle.dump(cyc_losses, file, pickle.HIGHEST_PROTOCOL)
            self._save_checkpoint(max_epoch)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU.")
    mp.spawn(
        main,
        args=(world_size,),
        nprocs=world_size,
    )  #
