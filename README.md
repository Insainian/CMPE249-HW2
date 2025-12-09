# CMPE249-HW2


On my HPC I've followed the instructions of the mmdetection3d setup tutorial. All the demo scripts ran, but were originally failing due to me using the prebuilt MMCV wheels the tutorial told me to download. Instead I uninstalled that and ran the following:

```bash
conda activate mmdetection3d

pip uninstall -y mmcv mmcv-full mmcv-lite

cd ~
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git fetch --tags
git checkout v2.1.0

module load nvhpc-hpcx-cuda12/24.11

export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6
export MATH_LIB_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/12.6/targets/x86_64-linux

export CPATH=$CPATH:$CUDA_HOME/targets/x86_64-linux/include:$MATH_LIB_HOME/include
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib:$MATH_LIB_HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/targets/x86_64-linux/lib:$MATH_LIB_HOME/lib
export PATH=$CUDA_HOME/bin:$PATH

# Use GCC, not NVHPC's nvc++
export CC=$(command -v gcc)
export CXX=$(command -v g++)

export TORCH_CUDA_ARCH_LIST="6.0 8.0"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1

pip uninstall -y numpy
pip install 'numpy<2'

rm -rf build/ mmcv.egg-info/

python setup.py build_ext --inplace
pip install -v . --no-build-isolation --no-deps
```

Once that demo script (python demo/pcd_demo.py demo/data/kitti/000008.bin) worked, I had to regenerate .pkl files for nuscenes with:

```bash
cd ~/mmdetection3d
conda activate mmdetection3d

# (optional but clean) remove any old info files if they exist
rm -f data/nuscenes/nuscenes_infos_*.pkl

PYTHONPATH=. python tools/create_data.py nuscenes \
--root-path ./data/nuscenes \
--out-dir ./data/nuscenes \
--extra-tag nuscenes
```

The shared class folder on the HPC had an empty sweeps folder so I used my local one. The kitti dataset in the shared folder was fine however.

## Running Eval and Inference

I ran eval and inference using the following combinations:
### kitti + 3DSSD

Ran eval with this command (similar to all the other combinations) 
```bash
python simple_infer_main.py   --config     $M3D/modelzoo_mmdetection3d/3dssd_4x4_kitti-3d-car.py   --checkpoint $M3D/modelzoo_mmdetection3d/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth   --dataroot   $M3D/data/kitti   --dataset    kitti   --data-source cfg   --ann-file   $M3D/data/kitti/kitti_infos_val.pkl   --eval --eval-backend runner   --device cuda   --out-dir   $HWREPO/results/kitti_3dssd_eval   > $HWREPO/logs/kitti_3dssd_eval.log 2>&1
```

Ran Inference with 
```bash
python simple_infer_main.py   --config     $M3D/modelzoo_mmdetection3d/3dssd_4x4_kitti-3d-car.py   --checkpoint $M3D/modelzoo_mmdetection3d/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth   --dataroot   $M3D/data/kitti   --dataset    kitti   --data-source cfg   --ann-file   $M3D/data/kitti/kitti_infos_val.pkl   --max-samples 200   --device cuda   --no-open3d   --out-dir   $HWREPO/results/kitti_3dssd_viz   > $HWREPO/logs/kitti_3dssd_viz.log 2>&1
```

### kitti + PointPillars

Ran eval with 
```bash
python simple_infer_main.py   --config     $M3D/modelzoo_mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py   --checkpoint $M3D/modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth   --dataroot   $M3D/data/kitti   --dataset    kitti   --ann-file   $M3D/data/kitti/kitti_infos_val.pkl   --data-source cfg   --eval --eval-backend runner   --device cuda   --out-dir   $HWREPO/results/kitti_pointpillars_eval   > $HWREPO/logs/kitti_pointpillars_eval.log 2>&1
```
Ran inference with 
```bash
python simple_infer_main.py   --config     $M3D/modelzoo_mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py   --checkpoint $M3D/modelzoo_mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth   --dataroot   $M3D/data/kitti   --dataset    kitti   --data-source cfg   --max-samples 200   --device cuda   --no-open3d   --out-dir   $HWREPO/results/kitti_pointpillars_viz   > $HWREPO/logs/kitti_pointpillars_viz.log 2>&1
```

### nuscenes + PointPillars

Ran eval with 
```bash
python simple_infer_main.py   --config     $M3D/modelzoo_mmdetection3d/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py   --checkpoint $M3D/modelzoo_mmdetection3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth   --dataroot   $M3D/data/nuscenes   --dataset    nuscenes   --data-source cfg   --ann-file   $M3D/data/nuscenes/nuscenes_infos_val.pkl    --eval --eval-backend runner   --device cuda   --out-dir   $HWREPO/results/nuscenes_pointpillars_eval   > $HWREPO/logs/nuscenes_pointpillars_eval.log 2>&1
```
Ran inference with
```bash
python simple_infer_main.py   --config     $M3D/modelzoo_mmdetection3d/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py   --checkpoint $M3D/modelzoo_mmdetection3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth   --dataroot   $M3D/data/nuscenes   --dataset    nuscenes   --data-source cfg   --max-samples 200   --device cuda   --no-open3d   --out-dir   $HWREPO/results/nuscenes_pointpillars_viz   > $HWREPO/logs/nuscenes_pointpillars_viz.log 2>&1
```

With this inference I had to run with using --no-save-image as otherwise I got a   File "/fs/atipa/home/015619422/CMPE249-HW2/detection3d/simple_infer_utils.py", line 2489, in inference_loop
paths, pts, meta['lidar2img'],
KeyError: 'lidar2img'

### nuscenes + CenterPoint

Ran eval using 
```bash
python simple_infer_main.py   --config     $M3D/modelzoo_mmdetection3d/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py   --checkpoint $M3D/modelzoo_mmdetection3d/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth   --dataroot   $M3D/data/nuscenes   --ann-file     $M3D/data/nuscenes/nuscenes_infos_val.pkl   --dataset    nuscenes   --data-source cfg   --eval --eval-backend runner   --device cuda   --out-dir   $HWREPO/results/nuscenes_centerpoint_eval   > $HWREPO/logs/nuscenes_centerpoint_eval.log 2>&1
```

Ran inference using
```bash
python simple_infer_main.py \
>   --config     $M3D/modelzoo_mmdetection3d/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py \
>   --checkpoint $M3D/modelzoo_mmdetection3d/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth \
>   --dataroot   $M3D/data/nuscenes \
>   --dataset    nuscenes \
>   --data-source cfg \
>   --max-samples 200 \
>   --device cuda \
>   --no-open3d \
>   --no-save-images \
>   --out-dir   $HWREPO/results/nuscenes_centerpoint_viz \
>   > $HWREPO/logs/nuscenes_centerpoint_viz.log 2>&1
```
Once I had the .ply files, ran open3d_view_saved_ply.py to visualize them. Combined the _points.ply and _pred.ply files in geom to see both in the same image.