# SEMDA

This GitHub repository implements the paper named **SemDA: Communication-Efficient Data Aggregation Through Distributed Semantic Transmission [ICASSP 2024]** based on **ns3-ai**.


* Paper: https://ieeexplore.ieee.org/abstract/document/10446003?casa_token=cq1B6t3xzfoAAAAA:6ZknCKj4BokgpUMEeSiQlSZArNOw8hSpJdZPFseYB5-B27NjBva7SoMACIVDiB96LojOcw0eCLc
* ns3-ai: https://github.com/hust-diangroup/ns3-ai

## Environment Setup
1. ns3-ai Environment Setup


2. 


## Code Preparation
1. Clone this repository at `scratch/SEMDA`
```bash
cd YOUR_NS3_DIRECTORY
git clone https://github.com/jk-hong-mnc/SEMDA.git scratch/SEMDA
```

## Code Execution
1. Build for SEMDA code in ns-3 base directory
```bash
cd YOUR_NS3_DIRECTORY
./waf configure
./waf build
```

2. After activating the conda virtual environment, run the simulation at `scratch/SEMDA`
```bash
conda activate ns3ai
cd YOUR_NS3_DIRECTORY
cd scratch/SEMDA
python3 run.py --image_number=10
```
