# LCDFormer





## Requirements

python.

torch-gpu.

## Data Preparation

Step1: Download datasets([PEMS03](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS03),[PEMS04](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS04),[PEMS07](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS07),[PEMS07(M)](https://github.com/wengwenchao123/DDGCRN?tab=readme-ov-file),[HZME_OUTFLOW](https://github.com/DrownFish19/CorrSTN/tree/master/data/HZME_OUTFLOW)).

Step2: Process raw data

```bash
python PrepareData.py
```

Step3: Generate DTW data

```bash
python create_dtw.py
```

## Train

```bash
python run.py
```

### Config

You can modify the parameters in the [configurations](/configurations/).

### Attention

When using PEMS07, please ensure that you have approximately 40GB of GPU memory.

If unable to run PrepareData.py, you can modify your virtual memory based on the error message.



### Cite

If you find the paper useful, please cite as following:
```bash
@article{LCDFormer,
title = {LCDFormer: Long-term correlations dual-graph transformer for traffic forecasting},
journal = {Expert Systems with Applications},
volume = {249},
pages = {123721},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.123721},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424005876},
author = {Jiongbiao Cai and Chia-Hung Wang and Kun Hu},
}
```



