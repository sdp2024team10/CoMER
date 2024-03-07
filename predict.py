#!/usr/bin/env python
import sys
import datetime
from comer.datamodule import vocab
from comer.lit_comer import LitCoMER
from torchvision.transforms import ToTensor
import torch
from PIL import Image
from IPython.display import display
ckpt = '../lightning_logs/version_0/checkpoints/epoch=151-step=57151-val_ExpRate=0.6365.ckpt'
#model = LitCoMER.load_from_checkpoint(ckpt, map_location="cpu")
model = LitCoMER.load_from_checkpoint(ckpt)
model = model.eval()
device = torch.device("cpu")
model = model.to(device)
before = datetime.datetime.now()
img = Image.open(sys.argv[1])
img = ToTensor()(img)
mask = torch.zeros_like(img, dtype=torch.bool)
hyp = model.approximate_joint_search(img.unsqueeze(0), mask)[0]
pred_latex = vocab.indices2label(hyp.seq)
print(pred_latex)
after = datetime.datetime.now()
#print(f"prediction time: {after - before}")

