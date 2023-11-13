dataset_path = "./CTP_Wires_Chargers_etc"
sample_trial = 50  # keep low to avoid dataloading bottleneck
sample_threshold = 0.1  # sample patches to have more than this standard deviation
resize_min = 600
crop_size = 512
bit_depth = 16
batch_size = 16
num_workers = 16  # set according to process on node

max_epochs = 100
ckpt_per = 50
base_learning_rate = 1e-4
max_iter = 20