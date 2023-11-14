dataset_path = "./CTP_Wires_Chargers_etc"
sample_trial = 50  # keep low to avoid dataloading bottleneck
sample_threshold = 0.1  # sample patches to have more than this standard deviation
resize_min = 256
crop_size = 128
bit_depth = 16
batch_size = 16
num_workers = 16  # set according to process on node

max_epochs = 1000
ckpt_per = 100
base_learning_rate = 1e-4

pretrain_weights12 = "./weights/g12unet_1000_loss0.0037.pt"
pretrain_weights21 = "./weights/g21unet_1000_loss0.0037.pt"
