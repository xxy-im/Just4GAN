import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "../../data/vangogh2photo"

res_nums = 9
batch_size = 8
num_workers = 8
lr = 2e-4
cyc_lambda = 10
id_lambda = 5

n_epochs = 200
decay_epoch = 100

output_dir = "../../output"
ckpt_dir = "../../weights"
inference_ckpt = "../../weights/G-checkpoint-021epoch.pth"
