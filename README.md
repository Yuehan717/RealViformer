# [ECCV 2024] RealViformer
### RealViformer: Investigating Attention for Real-World Video Super-Resolution
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-<COLOR>.svg)](https://arxiv.org/abs/<INDEX>)    
Yuehan Zhang, Angela Yao  
National University of Singapore

## Key Insights
In this paper, we focus on investigating spatial and channel attention under real-world VSR settings:
- we investigate the sensitivity of two attention mechanisms to degraded queries and compare them for temporal feature aggregation;
- we reveal the high channel covariance of channel attention outputs;
- to validate our findings, we derive RealViformer, a channel-attention-based Transformer for RWVSR, with simple but improved transformer block design.

## Results


## Installation
Instructions on how to set up the environment and dependencies required to run the code. Provide step-by-step commands:
```sh
# Clone the repository
git clone https://github.com/Yuehan717/RealViformer.git

# Navigate into the repository
cd RealViformer

# Install dependencies
pip install -r requirements.txt
```
## Datasets
- Training dataset: REDS; the degradation is added on-the-fly.
- Testing datasets:
  - Real-world datasets: VideoLQ, RealVSR
  - Synthetic datasets: REDS-test, UDM10; the degradation is synthesized with the same degradation pipeline in training.
## Usage
As RealViformer focuses on architecture design, we only provide testing scripts. The pretrained model is available [here]().
```sh
```


