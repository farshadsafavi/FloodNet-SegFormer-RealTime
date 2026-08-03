# FloodNet SegFormer Real-Time Segmentation

Clean, reusable SegFormer code derived from the experimental notebooks associated with:

> F. Safavi and M. Rahnemoonfar, “Comparative Study of Real-Time Semantic Segmentation Networks in Aerial Images During Flooding Events,” *IEEE JSTARS*, 2023. https://doi.org/10.1109/JSTARS.2022.3219724

This folder is a **cleaned reimplementation**, not an archival copy of the exact original training environment. It supports SegFormer-B0 through B5 by changing `model_name` (for example, `nvidia/mit-b0` or `nvidia/mit-b3`).

This repository focuses on the SegFormer experiments from the 2023 real-time benchmark. For the U-Net models with MobileNetV2 and MobileNetV3 encoders used in both this study and our earlier 2021 comparison, see the companion repository:

> [FloodNet-MobileNet-Segmentation](https://github.com/farshadsafavi/FloodNet-MobileNet-Segmentation)

## Install

```bash
pip install -r requirements.txt
```

Download FloodNet separately. The expected layout is:

```text
FloodNet/
├── images/
│   └── 0001.jpg
└── masks/
    └── 0001_lab.png
```

## Train SegFormer-B3

```bash
python floodnet_segformer.py \
  --image-dir /path/to/FloodNet/images \
  --mask-dir /path/to/FloodNet/masks \
  --model-name nvidia/mit-b3 \
  --epochs 15
```

The script saves the reproducible split, configuration, metric history, and best Hugging Face checkpoint. See `segformer_quickstart.ipynb` for a short implementation example.

## Citation

```bibtex
@article{safavi2023comparative,
  author  = {Farshad Safavi and Maryam Rahnemoonfar},
  title   = {Comparative Study of Real-Time Semantic Segmentation Networks in Aerial Images During Flooding Events},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2023},
  doi     = {10.1109/JSTARS.2022.3219724}
}
```

Before publishing, add the repository license you want to use.
