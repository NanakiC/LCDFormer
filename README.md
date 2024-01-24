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



