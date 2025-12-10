# CMPE249-HW2

> 3D object detection eval/inference workflows for CMPE 249 HW2. Uses MMDetection3D framework with lightly modified inference script.
> 
## Repo Layout
- `detection3d/`: lightly modified `simple_infer_main.py` + helpers from the Professor Kaikai Liu's repo
- `logs/`: stdout/stderr from every evaluation and visualization run
- `results/`: benchmark JSON metrics in eval folder, and demo video + screenshots
- `open3d_view_saved_ply.py`: modified viewer script from the Professor Kaikai Liu's repo
- `mmdetection3d_env.yaml`: conda spec for reproducing the working environment
- `report.md`: concise 1–2 page write-up summarizing setup, metrics, visuals, and takeaways

## Environment Setup
### Quickstart (local workstation, container, or lab GPU node)
```bash
# clone both repos
# git clone <this repo>
# git clone https://github.com/open-mmlab/mmdetection3d.git $HOME/mmdetection3d

cd /path/to/CMPE249-HW2
conda env create -f mmdetection3d_env.yaml -n mmdetection3d_hw2
conda activate mmdetection3d_hw2

# (optional) register shortcuts used in the commands below
export HWREPO=$(pwd)
export M3D=$HOME/mmdetection3d
```
The YAML already pins PyTorch 2.9.1 + CUDA 12.6 wheels, MMCV 2.1.0, MMDet3D 1.4.0, Open3D 0.19.0, and auxiliary tooling (ffmpeg, notebook deps, etc.).

### HPC-specific MMCV build notes (A100 @ HPC1)
The pre-built MMCV wheels from the tutorial failed with CUDA 12.6 on HPC1. Building from source with NVHPC CUDA libraries fixed the issue:
```bash
conda activate mmdetection3d
pip uninstall -y mmcv mmcv-full mmcv-lite numpy
pip install 'numpy<2'

cd ~
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git fetch --tags && git checkout v2.1.0

module load nvhpc-hpcx-cuda12/24.11
export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6
export MATH_LIB_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux
export CPATH=$CPATH:$CUDA_HOME/targets/x86_64-linux/include:$MATH_LIB_HOME/include
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib:$MATH_LIB_HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib:$MATH_LIB_HOME/lib
export PATH=$CUDA_HOME/bin:$PATH
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export TORCH_CUDA_ARCH_LIST="6.0 8.0"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1

rm -rf build/ mmcv.egg-info/
python setup.py build_ext --inplace
pip install -v . --no-build-isolation --no-deps
```
Once `python demo/pcd_demo.py demo/data/kitti/000008.bin` succeeds inside `~/mmdetection3d`, the rest of the homework scripts work.

## Dataset Preparation
1. Stage datasets under `$M3D/data/` (shared class paths on HPC already contain KITTI and NuScenes).
2. For NuScenes, there was no sweeps files so had to use my locally downloaded NuScenes. Needed to regenerate the info files to avoid stale metadata:
```bash
cd $M3D
conda activate mmdetection3d
rm -f data/nuscenes/nuscenes_infos_*.pkl
PYTHONPATH=. python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir   ./data/nuscenes \
  --extra-tag nuscenes
```

## Running Evaluation & Inference
Set helper env vars for brevity:
```bash
export HWREPO=/fs/atipa/home/<student_id>/CMPE249-HW2
export M3D=/home/<student_id>/mmdetection3d
export CUDA_VISIBLE_DEVICES=0
```
General command template:
```bash
python detection3d/simple_infer_main.py \
  --config     <config.py> \
  --checkpoint <model_checkpoint.pth> \
  --dataroot   <dataset_root> \
  --dataset    <kitti|nuscenes> \
  --data-source cfg \
  --ann-file   <info.pkl> \
  [--eval --eval-backend runner | --max-samples 200] \
  --device cuda \
  [--no-open3d] [--no-save-images] \
  --out-dir <results_subdir>
```
Executed combinations (logs live in `logs/` and artifacts in `results/`):

| Dataset | Model | Config | Checkpoint | Eval out/log | Viz out/log |
| --- | --- | --- | --- | --- | --- |
| KITTI | 3DSSD | `3dssd_4x4_kitti-3d-car.py` | `3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth` | `results/kitti_3dssd_eval` / `logs/kitti_3dssd_eval.log` | `results/kitti_3dssd_viz` / `logs/kitti_3dssd_viz.log` |
| KITTI | PointPillars | `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py` | `hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth` | `results/kitti_pointpillars_eval` / `logs/kitti_pointpillars_eval.log` | `results/kitti_pointpillars_viz` / `logs/kitti_pointpillars_viz.log` |
| NuScenes | PointPillars | `pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py` | `hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth` | `results/nuscenes_pointpillars_eval` / `logs/nuscenes_pointpillars_eval.log` | `results/nuscenes_pointpillars_viz` / `logs/nuscenes_pointpillars_viz.log` |
| NuScenes | CenterPoint | `centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py` | `centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth` | `results/nuscenes_centerpoint_eval` / `logs/nuscenes_centerpoint_eval.log` | `results/nuscenes_centerpoint_viz` / `logs/nuscenes_centerpoint_viz.log` |

All inference runs used `--max-samples 200` to keep artifact folders manageable while still satisfying the ≥200 frame requirement for the stitched video.

## Visualization & Media Exports
1. Install Open3D locally.
2. View saved point clouds locally:
```bash
python open3d_view_saved_ply.py \
  --dir results/<*_viz> \
  --view-json   results/<*_viz>/view.json \
  --basename <frame#> # e.g., 0 for 0_point.ply and 0_pred.ply
```
3. To get the JSON, run without --view-json first to generate it. Move it around in the pop-up window to get a good view, then press H to see
   the commands. Usually it will be command/ctrl c to copy the view settings to clipboard. Paste it into a file named view.json
4. Build the demo video once the PNG frames exist (from `results/*_viz/frame`):
```bash
cd results/<*_viz>/frames
ffmpeg -framerate 7 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p demo.mp4
```

## Troubleshooting Notes
- NuScenes inference raises `KeyError: 'lidar2img'` when `--save-images` is enabled; rerun with `--no-save-images` or augment the metadata dict in `simple_infer_utils.py` before enabling captures.
- Reinstalling MMCV from source (instructions above) resolved CUDA 12.6 compatibility issues on HPC1.

Refer to `report.md` for the latest metrics table and qualitative analysis.
