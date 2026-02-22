
import os
from export_weights import export_model

models = [
    ('results/cifar10_20260219_182845/checkpoints/checkpoint_latest.pth', 'frontend/public/models/cifar10'),
    ('results/ffhq_64_20260220_183709/checkpoints/checkpoint_latest.pth', 'frontend/public/models/ffhq_64')
]

modes = ['f32', 'f16', 'q8', 'q4']

for ckpt, out_dir in models:
    if not os.path.exists(ckpt):
        print(f"Skipping missing checkpoint: {ckpt}")
        continue
    for mode in modes:
        try:
            export_model(ckpt, out_dir, mode)
        except Exception as e:
            print(f"Error exporting {ckpt} in {mode}: {e}")
