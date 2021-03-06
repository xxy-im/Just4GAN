import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "../../data/Anime Sketch Colorization Pair/data/train"
val_dir = "../../data/Anime Sketch Colorization Pair/data/val"

batch_size = 64
num_workers = 4
lr = 2e-4
l1_lambda = 100

epochs = 300

output_dir = "../../output"
ckpt_dir = "../../weights"
inference_ckpt = "../../weights/G-checkpoint-021epoch.pth"
