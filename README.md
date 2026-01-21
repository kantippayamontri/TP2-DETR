# TP²‑DETR

**Unlocking Deformable DETR for Zero‑Shot Temporal Action Proposal Generation with Temporal Feature Pyramids**

## Requirements

- **Python**: 3.8.20
- **PyTorch**: 2.0.1
- **CUDA**: 11.7
- **GPU**: Tested on a single RTX 4080 Super

## Project Structure

```
TP2_DETR/
├── README.md
├── ActivityNet13/          # ActivityNet dataset directory
├── Thumos14/               # Thumos14 dataset directory
├── GAP/                    # GAP baseline implementation
│   ├── main.py
│   ├── train.py
│   ├── test.py
│   ├── dataset.py
│   ├── options.py
│   ├── requirements.txt
│   ├── config/             # Configuration files for different settings
│   │   ├── ActivityNet13_CLIP_zs_50.yaml
│   │   ├── ActivityNet13_CLIP_zs_75.yaml
│   │   ├── Thumos14_CLIP_zs_50_8frame.yaml
│   │   ├── Thumos14_CLIP_zs_75_8frame.yaml
│   │   └── Thumos14_ViFi-CLIP_zs_75_8frame.yaml
│   ├── data/               # Dataset annotations and features
│   │   ├── ActivityNet13/
│   │   │   ├── ActivityNet13_annotations.json
│   │   │   ├── ActivityNet13_description_v*.json
│   │   │   └── CLIP/
│   │   └── Thumos14/
│   │       ├── Thumos14_annotations.json
│   │       ├── Thumos14_description_v*.json
│   │       └── CLIP/
│   ├── eval/               # Evaluation scripts
│   ├── models/             # Model implementations
│   │   ├── clip/
│   │   └── ConditionalDetr/
│   ├── splits/             # Train/test split files
│   │   ├── train_50_test_50/
│   │   └── train_75_test_25/
│   └── utils/              # Utility functions
└── TP2-DETR/               # Main TP²-DETR implementation
    ├── environment.yml
    ├── readme.md
    ├── train.sh
    ├── train_all.sh
    ├── train_all_ablation.sh
    ├── ActionFormer/       # ActionFormer-based architecture
    │   ├── main.py
    │   ├── train.py
    │   ├── test.py
    │   ├── dataset.py
    │   ├── options.py
    │   ├── actionformer.py
    │   ├── SAPM.py         # Semantic-Aware Proposal Module
    │   ├── VideoSA.py      # Video Self-Attention
    │   ├── blocks.py
    │   ├── blocks_causal.py
    │   ├── config/         # Configuration files
    │   ├── eval/           # Evaluation utilities
    │   ├── models/         # Model components
    │   └── utils/          # Helper functions
    └── Deformable-DETR/    # Deformable DETR baseline
        ├── main.py
        ├── engine.py
        ├── requirements.txt
        ├── configs/
        ├── datasets/
        ├── models/
        └── util/
```

## Setup

### 1. Create the Conda environment

```bash
conda env create -f TP2-DETR/environment.yml
conda activate tp2
```

### 2. Build Deformable‑DETR ops

```bash
# Navigate to the ops directory
cd TP2-DETR/ActionFormer/models/ops

# Build the custom CUDA extensions
bash make.sh
```

### Prerequisites ✅

Before building, make sure you have:

- CUDA 11.7 installed and the `CUDA_HOME` environment variable set.
- `gcc/g++` compiler installed.
- PyTorch 2.0.1 with CUDA support installed.
- The Conda environment activated (created from `environment.yml`).
