# LCDFormer

Brief description of your project.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Requirements

python.

torch-gpu.

## Data Preparation

Step1: Download datasets(PEMS03,PEMS04,PEMS07,PEMS07(M),HZME_OUTFLOW).

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

### config

You can modify the parameters in the [configurations](/configurations/).


