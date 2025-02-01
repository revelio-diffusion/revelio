### Set up
```
conda install --file revelio.yaml
conda activate revelio
```

### 1. Extract intermediate diffusion features 
Extracting SD 1.5 feature at timestep=25, bottleneck on Caltech-101 
```
python extract_feature.py --timestep 25 --block_name mid_block --dataset_name dpdl-benchmark/caltech101 
```

### 2. Train k-SAE with extracted features
```
python train_ksae.py 
