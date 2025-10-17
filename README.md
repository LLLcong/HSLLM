# HSLLM: hidden state-augmented prompt-based LLM for long-term time series forecasting in industrial process

## Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Run](#run)

## Installation
The code is tested with Python 3.11, PyTorch 2.0.1
```
conda create -n hsllm python=3.11
conda activate hsllm
pip install -r requirements.txt
```

## Dataset
The kiln DCS dataset is available at [link](https://pan.quark.cn/s/56ab099037e1).

Please contact cong_l.lc@connect.hku.hk to get the extraction code.

## Run

**Parameters setting**:

prompt include: statistics, learnable_zero, learnable_uniform, word_embedding, HS_embedding, no_conv, none

seq_len include: 120, 240, 360

pred_len include: 30, 60, 90, 120

gpt_layers include: 3, 6, 9, 12

### Training

```bash
conda activate hsllm
bash ./scripts/kiln.sh <device> <prompt> <seq_len> <pred_len> <gpt_layers>
# For example
bash ./scripts/kiln.sh 0 statistics 240 120 6
```

### Evaluation
```
bash ./scripts/kiln_test.sh <device> <prompt> <seq_len> <pred_len> <gpt_layers>
# For example
bash ./scripts/kiln_test.sh 0 HS_embedding 240 120 6
```




