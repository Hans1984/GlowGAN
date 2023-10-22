# [GlowGAN](https://glowgan.mpi-inf.mpg.de/resource/glowgan.pdf): GlowGAN: Unsupervised Learning of HDR Images from LDR Images in the Wild [ICCV 2023]


### Installation
Use the following commands with Anaconda to create and activate your environment:
  - ```conda env create -f environment.yml```
  - ```conda activate glow_gan```

If you encounter some setting problems, you may find solutions from [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl).

### Pretrained Models
You can download the pretrianed models from [Models](https://drive.google.com/drive/folders/1Wf4PTy3fFUOzf9RRFr4GowHJF-EHLMkt?usp=sharing).

### Training
We follow the progressive training be strategy, strat from the low resolution and increase the resolution step by step.

```
 python train.py --outdir=<OUT_DIR> --cfg=stylegan3-t -- 
  data=<DATA_DIR> --gpus=4 --batch=32 --mirror=1 --snap 20 --batch-gpu 8 --kimg 10000 --syn_layers 7
```
### Training the super-resolution stages
Continuing with pretrained stem:
```
python train.py --outdir=<OUT_UPSCALE_DIR> --cfg=stylegan3-t --data=<DATA_HIGH_RES_DIR> \
  --gpus=4 --batch=32 --mirror=1 --snap 20 --batch-gpu 8 --kimg 10000 --syn_layers 7 \
  --superres --up_factor 2 --head_layers 7 \
  --path_stem <PRETRAINED_MODEL_DIR>
```

### Image Generation
```
python gen_images.py --outdir=<OUT_DIR> --trunc=0.7 --seeds=449 --batch-sz 1 --network=<PRETRAINED_MODEL_DIR>
```

### Inverse Tone Mapping
```
python run_inversion.py --outdir=<OUT_DIR> --target data/001.png --inv-steps 1000 --run-pti --pti-steps 350 --network=<PRETRAINED_MODEL>
```
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{wang2023glowgan,
  title={GlowGAN: Unsupervised Learning of HDR Images from LDR Images in the Wild},
  author={Wang, Chao and Serrano, Ana and Pan, Xingang and Chen, Bin and Myszkowski, Karol and Seidel, Hans-Peter and Theobalt, Christian and Leimk{\"u}hler, Thomas},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10509--10519},
  year={2023}
}</code></pre>
  </div>
</section>

### Acknowledge
This source code is derived from the awesome GAN model [StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl). We really appreciate the contributions of the authors to that repository.



