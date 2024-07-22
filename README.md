# [ECCV 2024] RealViformer
### RealViformer: Investigating Attention for Real-World Video Super-Resolution
[![arXiv](https://img.shields.io/badge/arXiv-2407.13987-<COLOR>.svg)](https://arxiv.org/abs/2407.13987)    
Yuehan Zhang, Angela Yao  
National University of Singapore

## Key Insights
In this paper, we focus on investigating spatial and channel attention under real-world VSR settings:
- we investigate the sensitivity of two attention mechanisms to degraded queries and compare them for temporal feature aggregation;
- we reveal the high channel covariance of channel attention outputs;
- to validate our findings, we derive RealViformer, a channel-attention-based Transformer for RWVSR, with a simple but improved transformer block design.

## TODOs
- <del>Public the repository</del>
- Update links to datasets
- Add video results


## Installation
#### Set up environment
Python >= 3.9  
PyTorch > 1.12
#### Install RealViformer
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
As RealViformer focuses on architecture design, we only provide testing scripts. The pretrained model is available [here](https://drive.google.com/drive/folders/1UzDfFSy5oELl7Z-umF_QhMQhUbUU378y?usp=sharing).
```sh
python inference_realviformer.py --model_path pretrained_model/weights.pth --input_path [path to video folder] --save_path results/ --interval 100
```

## Acknowledgement
The code is based on [BasicVSR](https://github.com/ckkelvinchan/BasicVSR-IconVSR) and [Restormer](https://github.com/swz30/Restormer). Thanks to their great work!


