# SEMDA

This GitHub repository implements the paper named **SemDA: Communication-Efficient Data Aggregation Through Distributed Semantic Transmission [ICASSP 2024]** based on **ns3-ai**.


* Paper: https://ieeexplore.ieee.org/abstract/document/10446003?casa_token=cq1B6t3xzfoAAAAA:6ZknCKj4BokgpUMEeSiQlSZArNOw8hSpJdZPFseYB5-B27NjBva7SoMACIVDiB96LojOcw0eCLc
* ns3-ai: https://github.com/hust-diangroup/ns3-ai

## Environment Setup (Based on Linux Ubuntu 22.04.03)
**1. ns3-ai Installation**

* Install the prerequisite packages for ns-3
```bash
sudo apt-get update
sudo apt-get install g++ cmake ninja-build git ccache pkg-config sqlite3 qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools gir1.2-goocanvas-2.0 gir1.2-gtk-3.0 gdb valgrind tcpdump libsqlite3-dev libc6-dev sqlite sqlite3
```

* Download ns3 version of 3.35 and extract it to your `home` folder: https://www.nsnam.org/releases/ns-allinone-3.35.tar.bz2

* Download ns3-ai repository from **master(!)** branch: https://github.com/hust-diangroup/ns3-ai/tree/master
  * **Do not download via the git clone command, but be sure to download it in ZIP file format from the GitHub link!**
  * Unzip the downloaded file and change the extracted folder name from ns3-ai-master to **ns3-ai**
  * Move the ns3-ai folder to `ns-3.35/contrib/ns3-ai`

**2. Anaconda Virtual Environment Setup (Highly recommended!!!!)**

Install Anaconda
```bash
sudo apt-get install curl -y
curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
sha256sum anaconda.sh
```

```bash
bash anaconda.sh
```
After entering the `bash` command above, perform the following steps:
* Keep pressing `ENTER` until you read all the license terms
* `Do you accept the license terms? [yes|no]` >>> Type `yes` and press `ENTER`
* `[/home/ns3/anaconda3]` >>> Just ㅔress `ENTER`
* `Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]` >>> Type `yes` and press `ENTER`

```bash
sudo vi ~/.bashrc
```
After entering the `vi` command above, perform the following steps:
* Press `a` at the last line of the code
* Type `export PATH=~/anaconda3/bin:~/anaconda3/condabin:$PATH`
* Press `ESC` button
* Type `:wq` and press `ENTER`

```bash
source ~/.bashrc
```

Create and activate `ns3ai` virtual environment **(Make sure the ns3ai virtual environment is always active!!)**
```bash
conda create –n ns3ai python=3.8
conda activate ns3ai
```

Install ns3ai as a Python library at your `ns3ai` virtual environment
```bash
cd ns-allinone-3.35/ns-3.35/contrib/ns3-ai/py_interface
pip3 install . --user
pip3 install . e
pip3 install . 
```

**3. Build ns-3**

Run the commands below in your `ns-3.35` base directory
```bash
./waf configure
./waf build
```

## Code Preparation
**1. Clone this repository at `scratch/SEMDA`**
```bash
cd YOUR_NS3_DIRECTORY
git clone https://github.com/jk-hong-mnc/SEMDA.git scratch/SEMDA
```

**2. Install the required Python library**
```bash
pip3 install torch torchvision numpy
```

## Code Execution
**1. Modify the directory of image and model weights to match the path in your Ubuntu environment**

* `SEMDA/sim.cc` line 315-316
```bash
std::string imagePath_n0 = "/home/ns3ai/ns-allinone-3.35/ns-3.35/scratch/SEMDA/data/test_dataset.bin";
std::string imagePath_n1 = "/home/ns3ai/ns-allinone-3.35/ns-3.35/scratch/SEMDA/data/test_dataset_transformed.bin";
```

* `SEMDA/run.py` line 189
```bash
model_dir = "/home/ns3ai/ns-allinone-3.35/ns-3.35/scratch/SEMDA/weight/"
```

**2. Build for SEMDA code in ns-3 base directory**
```bash
cd YOUR_NS3_DIRECTORY
./waf configure
./waf build
```

**3. After activating the conda virtual environment, run the simulation at `scratch/SEMDA` **(Only one terminal is required to run the simulation.)****
```bash
conda activate ns3ai
cd YOUR_NS3_DIRECTORY
cd scratch/SEMDA
python3 run.py --image_number=10
```
