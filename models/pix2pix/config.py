import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "../../data/Anime Sketch Colorization Pair/train"
val_dir = "../../data/Anime Sketch Colorization Pair/val"

batch_size = 16
lr = 2e-4
l1_lambda = 100

epochs = 500

output_dir = "../../output"
ckpt_dir = "../../weights"
inference_ckpt = ""
