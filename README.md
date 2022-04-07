##  :sound: Sound-guided Semantic Image Manipulation
Official Pytorch Implementation

![Teaser image](https://kr.object.ncloudstorage.com/cvpr2022/overview.png)


**Sound-guided Semantic Image Manipulation**<br>

Paper : https://arxiv.org/abs/2112.00007 <br>
Project Page: https://kuai-lab.github.io/cvpr2022sound/ <br>
Seung Hyun Lee, Wonseok Roh, Wonmin Byeon, Sang Ho Yoon, Chanyoung Kim, Jinkyu Kim*, and Sangpil Kim* <br>

Abstract: *The recent success of the generative model shows that leveraging the multi-modal embedding space can manipulate an image using text information. However, manipulating an image with other sources rather than text, such as sound, is not easy due to the dynamic characteristics of the sources. Especially, sound can convey vivid emotions and dynamic expressions of the real world. Here, we propose a framework that directly encodes sound into the multi-modal~(image-text) embedding space and manipulates an image from the space. Our audio encoder is trained to produce a latent representation from an audio input, which is forced to be aligned with image and text representations in the multi-modal embedding space. We use a direct latent optimization method based on aligned embeddings for sound-guided image manipulation. We also show that our method can mix different modalities, i.e., text and audio, which enrich the variety of the image modification. The experiments on zero-shot audio classification and semantic-level image classification show that our proposed model outperforms other text and sound-guided state-of-the-art methods.*

## :floppy_disk: Installation
For all the methods described in the paper, is it required to have:
- Anaconda
- [CLIP](https://github.com/openai/CLIP)

Specific requirements for each method are described in its section. 
To install CLIP please run the following commands:
  ```shell script
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```

## :hammer: Method
![Method image](https://kr.object.ncloudstorage.com/cvpr2022/main_figure.png)

### 1. CLIP-based Contrastive Latent Representation Learning.
**Dataset Curation.**

We create an audio-text pair dataset with the vggsound dataset. We also used the audioset dataset as the script below.

1. Please download [`vggsound.csv`](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) from the link.
2. Execute `download.py` to download the audio file of the vggsound dataset.
2. Execute `curate.py` to preprocess the audio file (wav to mel-spectrogram).
```
cd soundclip
python3 download.py
python3 curate.py
```
**Training.**
```
python3 train.py
```

### 2. Sound-Guided Image Manipulation.
**Direct Latent Code Optimization.**

The code relies on the StyleCLIP pytorch implementation. 

- [pretrained-StyleGAN2](https://kr.object.ncloudstorage.com/cvpr2022/landscape.pt)
- [pretrained-audio encoder](https://kr.object.ncloudstorage.com/cvpr2022/resnet18_57.pth)

```
python3 optimization/run_optimization.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/explosion.wav" --save_source_image_path "./source_image" --save_manipulated_image_path "./manipulated_image" --ckpt ./pretrained_models/landscape.pt --stylegan_size 256"
```
## :golf: Results

### Zero-shot Audio Classification Accuracy.
|Model| Supervised Setting  | Zero-Shot  |  ESC-50 |  UrbanSound 8K |
|:-:|:-:|:-:|:-:|:-:|
| ResNet50  |  :white_check_mark: |  - | 66.8%  | 71.3%  |
|  Ours (Without Self-Supervised) | -  | -  | 58.7%  | 63.3%  |
|  :sparkles: Ours (Logistic Regression) |  - |  - |  72.2% |  66.8% |
|  Wav2clip |  - |  :white_check_mark: |  41.4% | 40.4%  |
| AudioCLIP  | - | :white_check_mark:  | 69.4%  | 68.8%  |
| Ours (Without Self-Supervised)  |  - |  :white_check_mark: |  49.4% | 45.6%  |
| :sparkles: Ours  |  - |  :white_check_mark: | 57.8%  | 45.7%  |



### Manipulation Results.

**LSUN.**
![LSUN image](https://kr.object.ncloudstorage.com/cvpr2022/figure4_submission.png)

### About StyleGAN3
The code is borrowed from the link below.

Link : https://colab.research.google.com/github/ouhenio/StyleGAN3-CLIP-notebook/blob/main/StyleGAN3%2BCLIP.ipynb

StyleGAN3 + Our CLIP-based Sound Representation
```python

import sys

import io
import os, time, glob
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import clip
import unicodedata
import re
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from einops import rearrange
from collections import OrderedDict

import timm
import librosa
import cv2
    
class AudioEncoder(torch.nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv = torch.nn.Conv2d(1, 3, (3, 3))
        self.feature_extractor = timm.create_model("resnet18", num_classes=512, pretrained=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.feature_extractor(x)
        return x

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def run(timestring):
  torch.manual_seed(seed)

  # Init
  with torch.no_grad():
    qs = []
    losses = []
    for _ in range(8):
      q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
      images = G.synthesis(q * w_stds + G.mapping.w_avg)
      embeds = embed_image(images.add(1).div(2))
      loss = prompts_dist_loss(embeds, targets, spherical_dist_loss).mean(0)
      i = torch.argmin(loss)
      qs.append(q[i])
      losses.append(loss[i])
    qs = torch.stack(qs)
    losses = torch.stack(losses)
    i = torch.argmin(losses)
    q = qs[i].unsqueeze(0).requires_grad_()

  w_init = (q * w_stds + G.mapping.w_avg).detach().clone()
  # Sampling loop
  q_ema = q
  opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0,0.999))
  loop = tqdm(range(steps))
  for i in loop:
    opt.zero_grad()
    w = q * w_stds + G.mapping.w_avg
    image = G.synthesis(w , noise_mode='const')
    embed = embed_image(image.add(1).div(2))
    loss = 0.1 *  prompts_dist_loss(embed, targets, spherical_dist_loss).mean() + ((w - w_init) ** 2).mean()
    # loss = prompts_dist_loss(embed, targets, spherical_dist_loss).mean()
    loss.backward()
    opt.step()
    loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())

    q_ema = q_ema * 0.9 + q * 0.1

    final_code = q_ema * w_stds + G.mapping.w_avg
    final_code[:,6:,:] = w_init[:,6:,:]
    image = G.synthesis(final_code, noise_mode='const')

    if i % 10 == 9 or i % 10 == 0:
      # display(TF.to_pil_image(tf(image)[0]))
      print(f"Image {i}/{steps} | Current loss: {loss}")
      pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0,1).cpu())
      os.makedirs(f'samples/{timestring}', exist_ok=True)
      pil_image.save(f'samples/{timestring}/{i:04}.jpg')


device = torch.device('cuda:0')
print('Using device:', device, file=sys.stderr)

model_url = "./pretrained_models/stylegan3-r-afhqv2-512x512.pkl"

with open(model_url, 'rb') as fp:
  G = pickle.load(fp)['G_ema'].to(device)

zs = torch.randn([100000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)

m = make_transform([0,0], 0)
m = np.linalg.inv(m)
G.synthesis.input.transform.copy_(torch.from_numpy(m))
audio_paths = "./audio/dog-sad.wav"
steps = 200
seed = 2474

audio_paths = [frase.strip() for frase in audio_paths.split("|") if frase]

clip_model = CLIP()
audio_encoder = AudioEncoder()
audio_encoder.load_state_dict(copyStateDict(torch.load("./pretrained_models/resnet18.pth", map_location=device)))

audio_encoder = audio_encoder.to(device)
audio_encoder.eval()

targets = []
n_mels = 128
time_length = 864
resize_resolution = 512

for audio_path in audio_paths:
    y, sr = librosa.load(audio_path, sr=44100)
    audio_inputs = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    audio_inputs = librosa.power_to_db(audio_inputs, ref=np.max) / 80.0 + 1

    zero = np.zeros((n_mels, time_length))
    h, w = audio_inputs.shape
    if w >= time_length:
        j = (w - time_length) // 2
        audio_inputs = audio_inputs[:,j:j+time_length]
    else:
        j = (time_length - w) // 2
        zero[:,:w] = audio_inputs[:,:w]
        audio_inputs = zero
    
    audio_inputs = cv2.resize(audio_inputs, (n_mels, resize_resolution))
    audio_inputs = np.array([audio_inputs])
    audio_inputs = torch.from_numpy(audio_inputs.reshape((1, 1, n_mels, resize_resolution))).float().to(device)
    with torch.no_grad():
        audio_embedding = audio_encoder(audio_inputs)
        audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
    targets.append(audio_embedding)

timestring = time.strftime('%Y%m%d%H%M%S')
run(timestring)
```