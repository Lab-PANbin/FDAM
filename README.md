# FDAM: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization

PyTorch implementation of FDAM for joint box detection and grid classification of thin and tiny pavement cracks.

**Haiyang Xu, Bin Pan, Mengze Wang, Qiaoying Qu, and Zhenwei Shi**
[Paper](https://arxiv.org/abs/2611.19089) · [PDF](https://arxiv.org/pdf/2611.19089) · [Code](https://github.com/Lab-PANbin-owner/FDAM)

## Repository contents

```text
configs/fdam/       Dataset-specific training and checkpoint configurations
engine/            Models, data loading, losses, training, and evaluation
tools/inference/  Grid export, offline metrics, and visualization
train.py           Training and evaluation entry point
environment.yml    Conda environment specification
requirements.txt   Python dependencies
```

This source release excludes datasets, weights, experiment results, logs, cluster launch scripts, deployment artifacts, and standalone debugging code. Evaluation utilities remain included so that predictions can be measured. Generated outputs are written locally and ignored by Git.

## Abstract

Pavement crack grid localization remains challenging for thin and tiny cracks due to weak visual cues and background interference. Box detection often misses these cracks, while supervised semantic segmentation faces limited pixel-wise annotations and high labeling costs. In this paper, we propose a Frequency-Direction-Aware Multi-Task framework (FDAM) with box detection and grid classification branches. The core grid classification branch comprises Frequency-Domain Enhancement (FDE) and Directional Feature Refinement (DFR). FDE adaptively weights spectral bands and fuses reconstructed features with spatial features. DFR captures thin crack patterns through multi-scale strip convolutions, direction-sensitive modulation, and feature recalibration. Sparse positive cells and ambiguous local predictions motivate adaptive grid supervision. Adaptive Importance Reweighting emphasizes uncertain predictions and local label transitions, with focal supervision for all cells. Our contribution is to couple the box detection and grid classification branches through shared feature learning for pavement crack grid localization. We validate FDAM on our RoadDistress2026 dataset of thin and tiny cracks and further evaluate it on public datasets Crack500 and DeepCrack. The code is available at https://github.com/Lab-PANbin-owner/FDAM.

## Method

A shared backbone and encoder feed the detection and grid branches. FDE adaptively weights spectral bands, DFR captures directional crack structure, and adaptive importance reweighting emphasizes uncertain cells and local label transitions. At inference, class-aware box-grid association retains grid predictions that overlap a detection of the same category.

The main implementation is in `engine/head/frequency_attention_grid_head.py`, `engine/head/direction_aware_grid_head.py`, and `engine/deim/deim_criterion.py`.

## Installation

Linux with an NVIDIA GPU is recommended. Run all commands from the repository root.

```bash
git clone https://github.com/Lab-PANbin-owner/FDAM.git
cd FDAM
conda env create -f environment.yml
conda activate fdam
```

The environment uses Python 3.10, PyTorch 2.2.2, and CUDA 12.1. Install an NVIDIA driver compatible with CUDA 12.1 before using the GPU.

## Data and pretrained weights

Organize each dataset as follows, or edit its paths in `configs/fdam/`:

```text
data/<dataset>/
  images/{train,val,test}/
  annotations/instances_train.json
  annotations/instances_val.json
  annotations/instances_test.json
```

The dataset names are `roaddistress2026`, `crack500`, and `deepcrack`. Annotations use COCO image records and pixel-coordinate boxes `[x, y, width, height]`. Detection class IDs are zero-based (`0` to `C-1`). Separately annotated grid regions for class `c` use category ID `-(c+1)`. Include `id`, `image_id`, `category_id`, `bbox`, `area`, and `iscrowd` in each annotation and list both annotation families in `categories`. Detection boxes and grid regions are separate labels; the loader does not generate one from the other or convert segmentation masks automatically.

Download the corresponding backbone initialization from the [DEIMv2 pretrained-weight guide](https://github.com/Intellindust-AI-Lab/DEIMv2) and place it in `ckpts/`:

| Dataset | Configuration | Backbone file |
| --- | --- | --- |
| RoadDistress2026 | `configs/fdam/roaddistress2026.yml` | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| Crack500 | `configs/fdam/crack500.yml` | `vitt_distill.pt` |
| DeepCrack | `configs/fdam/deepcrack.yml` | `vittplus_distill.pt` |

These three training recipes enable FDE, DFR, and AIR. The two `*_checkpoint.yml` configurations preserve the existing RoadDistress2026/Crack500 AIR-off settings for their archived checkpoints. Match the configuration to the run being evaluated. Datasets, backbone weights, and trained FDAM checkpoints are not included in this repository. Supply the required backbone file before training; without it, the current backbone loader falls back to random initialization.

## Training

```bash
# Two GPUs. Change the config to select another dataset.
torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/fdam/deepcrack.yml --seed 0
```

Use `python train.py` for a single GPU. Add `-t ckpts/deimv2_pretrained.pth` to initialize from a matching full-detector checkpoint, or `-r outputs/<run>/last.pth` to resume. Do not combine `-t` and `-r`. Add `--use-amp` to enable mixed precision; otherwise the CLI uses full precision. `-u key=value` overrides configuration fields, for example `-u train_dataloader.total_batch_size=4`.

## Evaluation

For Crack500 and DeepCrack, the following exports box metrics and grid predictions. Evaluation reads `val_dataloader`, so override both paths to use the test split. Grid export uses one process.

```bash
python train.py -c configs/fdam/deepcrack.yml \
  -r ckpts/fdam_deepcrack.pth --test-only --output-dir outputs/deepcrack_test \
  -u grid_export=true use_ema=true \
  val_dataloader.dataset.img_folder=./data/deepcrack/images/test \
  val_dataloader.dataset.ann_file=./data/deepcrack/annotations/instances_test.json

python tools/inference/eval_grid_comparison.py \
  --manifest outputs/deepcrack_test/grid_predictions/manifest.json \
  --split test --grid-threshold 0.45 --det-threshold 0.30 \
  --require-box-metrics --output-dir outputs/deepcrack_test/grid_metrics
```

The example uses fixed thresholds. For a new run, select thresholds on validation with `--split val --sweep`, then pass its `selection.json` using `--selection` for test evaluation.

RoadDistress2026 uses the separate multi-class exporter and evaluator:

```bash
python tools/inference/export_road_grid_predictions.py \
  --backend fdam --model-root . --config configs/fdam/roaddistress2026_checkpoint.yml \
  --checkpoint ckpts/fdam_roaddistress2026.pth --model-source ema \
  --coco data/roaddistress2026/annotations/instances_val.json \
  --image-root data/roaddistress2026/images/val --split val --output-dir outputs/road_val

python tools/inference/eval_road_grid_protocol.py \
  --manifest outputs/road_val/manifest.json \
  --coco data/roaddistress2026/annotations/instances_val.json \
  --split val --sweep --output-dir outputs/road_val/metrics
```

For RoadDistress2026 test evaluation, export the test split and replace `--sweep` with `--selection outputs/road_val/metrics/selection.json`. Keep the same checkpoint and weight source. The evaluator reports LC/TC by default. Do not use the single-class `grid_export` path for this dataset.

## Visualization

The joint visualizer requires CUDA and supports all three datasets through their annotation category maps. It uses the input size from the configuration.

```bash
python tools/inference/torch_inf_road_vis_en.py \
  -c configs/fdam/deepcrack.yml -r ckpts/fdam_deepcrack.pth \
  -d data/deepcrack/images/test -o outputs/deepcrack_vis \
  --annotations data/deepcrack/annotations/instances_test.json \
  --model-source ema --det-thrh 0.30 --grid-thrh 0.45 \
  --save-det --save-grid --save-matched
```

This saves boxes, raw grid predictions, and associated grid predictions separately. Choose the checkpoint's actual weight source with `--model-source model` or `ema`.

## Acknowledgments

FDAM builds on [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), [DEIM](https://github.com/Intellindust-AI-Lab/DEIM), and [DINOv3](https://github.com/facebookresearch/dinov3). We thank the authors of these three projects for their valuable work and for sharing their implementations with the research community.

## Citation

If you use FDAM in your research, please cite our arXiv preprint:

```bibtex
@misc{xu2026fdam,
  title = {{FDAM}: A Frequency-Direction-Aware Multi-Task Framework for Pavement Crack Grid Localization},
  author = {Haiyang Xu and Bin Pan and Mengze Wang and Qiaoying Qu and Zhenwei Shi},
  year = {2026},
  eprint = {2611.19089},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2611.19089}
}
```
