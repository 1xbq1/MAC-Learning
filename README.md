# MAC-Learning
Multi-Granularity Anchor-Contrastive Representation Learning for Semi-supervised Skeleton-based Action Recognition
## Requirements
- python == 3.8.3
- pytorch == 1.11.0
- CUDA == 11.2
## Data Preparation
Download the raw data of [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D), [NW-UCLA](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0), and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn).
Then the commands for data preprocessing are as follows,
```
python ./data_gen/ntu_gendata.py
python ./data_gen/ucla_gendata.py
python ./data_gen/kinetics_gendata.py
```
## Training
- On NTU RGB+D cross-subject benchmark.
```
python main.py --config ./config/nturgbd-cross-subject/train_joint_aagcn.yaml
```
- On NTU RGB+D cross-view benchmark.
```
python main.py --config ./config/nturgbd-cross-view/train_joint_aagcn.yaml
```
- On NW-UCLA.
```
python main.py --config ./config/ucla/train_joint_aagcn.yaml
```
- On Skeleton-Kinetics.
```
python main.py --config ./config/kinetics-skeleton/train_joint_aagcn.yaml
```
## Trained model
The trained model will be uploaded as soon as possible.
## Citation
