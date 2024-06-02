# LatentMan : Generating Consistent Animated Characters using Image Diffusion Models
<center>
<h3>CVPRW 2024</h3>
</center>
<a href='https://abdo-eldesokey.github.io/latentman/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2312.07133'><img src='https://img.shields.io/badge/ArXiv-2312.03047-red'></a> 
<img src="https://raw.githubusercontent.com/abdo-eldesokey/text2ac-zero/gh-page/static/images/method.jpg" alt="method">
</div>

## Environment
This code was tested on Python 3.8 with Cuda 12.1 and PyTorch 2.3.

- To setup up the environemnt, simply create a Conda environemnt by running:

```
conda env create -f environment.yml
```

- Follow the instructions for parts 2,3 in MDM's [README.md](https://github.com/GuyTevet/motion-diffusion-model/tree/main) to download the required files for Motion-Diffusion-Model code located in `external/MDM`.

- You can download some examples of generated motions to `workspace` by running:
```
gdown "1IdaCPpRWrmRX5AVXXUXymwFVHXt2CNnW&confirm=t"
unzip workspace.zip
```

## Getting Started
Please refer to `getting_started.ipynb`.

## Acknowledgements 
Parts of this code base are adapted from [MDM](https://github.com/GuyTevet/motion-diffusion-model/tree/main), [Detectron2](https://github.com/facebookresearch/detectron2), and [MvDeCor](https://github.com/nv-tlabs/MvDeCor).

## Citation
If you use this code or parts of it, please cite our paper:
```

```
