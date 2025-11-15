# Environment
```
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Clone openvla-oft repo and pip install to download dependencies
git clone --recurse-submodules https://github.com/Boltzmachine/openvla-oft.git
cd openvla-oft
pip install -e .

pip uninstall transformers #! VERY IMPORTANT
pip show transformers # Make sure there are no transformers


cd transformers
pip install -e .
cd ..

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
MAX_JOBS=4 pip install "flash-attn==2.5.5" --no-build-isolation

cd vla-utils
pip install -r requirements.txt
cd ..
```

# Dataset
Download from [Huggingface](https://huggingface.co/datasets/openvla/modified_libero_rlds). Rename it to `dataset`.
The path should look like, for example, `dataset/libero_spatial_no_noops`.

# Training
```
bash tinglin/train.sh
```
Read the script to adjust the hyperparameters.

# Evaluation
Install extra libraries
```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From openvla-oft base dir
```

Modify the checkpoint path and run
```
bash tinglin/eval.sh
```

