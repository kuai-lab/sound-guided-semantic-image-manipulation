##  :sound: Sound-guided Semantic Image Manipulation (CVPR2022)
Official Pytorch Implementation

![Teaser image](https://kr.object.ncloudstorage.com/cvpr2022/overview.png)


**Sound-guided Semantic Image Manipulation**<br>
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2022

Paper : [CVPR 2022 Open Access](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Sound-Guided_Semantic_Image_Manipulation_CVPR_2022_paper.pdf) <br>
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
python3 optimization/run_optimization.py --lambda_similarity 0.002 --lambda_identity 0.0 --truncation 0.7 --lr 0.1 --audio_path "./audiosample/explosion.wav" --ckpt ./pretrained_models/landscape.pt --stylegan_size 256
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

**FFHQ.**
![FFHQ image](https://kr.object.ncloudstorage.com/cvpr2022/figure5_submission.png)

To see more diverse examples, please visit our [project page](https://kuai-lab.github.io/cvpr2022sound/)!  

## Citation
```
@InProceedings{Lee_2022_CVPR,
    author    = {Lee, Seung Hyun and Roh, Wonseok and Byeon, Wonmin and Yoon, Sang Ho and Kim, Chanyoung and Kim, Jinkyu and Kim, Sangpil},
    title     = {Sound-Guided Semantic Image Manipulation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3377-3386}
}
```
