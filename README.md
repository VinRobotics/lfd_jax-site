# lfd_jax: Learning from Demonstration Algorithms in Jax

<div align="center">

[<img src="assets/logo.png" width="200">]()

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![Linux platform](https://img.shields.io/badge/ubuntu-22.04-red)](https://releases.ubuntu.com/22.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-ApacheV2-yellow.svg)](https://opensource.org/license/apache-2-0/)

</div>

`lfd_jax` is a software package developed by [VinRobotics](https://vinrobotics.net/) that provides implementations of common Learning from Demonstration (LfD) algorithms in [Jax](https://github.com/jax-ml/jax) for robotic applications. The package is designed to be modular and extensible, making it easy to develop, test, and integrate new learning algorithms into robotic systems. For API documentation, please visit the [documentation](https://vinrobotics.github.io/lfd_jax/).

## Installation

Create a virtual environment with Python 3.10 and activate it:

```bash
# Create and activate virtual environment with Python 3.10
pip install uv
uv venv lfd_jax_env --python=3.10
source lfd_jax_env/bin/activate
```

Install:

```bash
git clone --recursive https://github.com/VinRobotics/lfd_jax.git

cd lfd_jax
uv pip install -e .

# Install lerobot in editable mode
cd third_party/lerobot
uv pip install -e ".[aloha, pusht]"

# Apply the patch
patch -p1 < ../../third_party_patches/lerobot.patch
```

(Optional) Install formaters (then staged files will be checked when doing code commits):

```bash
uv pip install ruff isort black
uv pip install pre-commit
pre-commit install

# run formatters once
pre-commit run --all-files
```

## Examples

```bash
cd ../../lfd_jax

# Behavior Cloning (BC) using LeRobot Push-T datasets (pushT_keypoint/pushT_image)
python scripts/train.py algo=bc_lerobot env=pushT_keypoint dataset_type=lerobot wandb.enable=true training.steps=200000 dataset.keep_video_in_ram=true training.eval_freq=1000 training.log_freq=1000

# Diffusion Policy (DP) using LeRobot Push-T datasets (pushT_keypoint/pushT_image)
python scripts/train.py algo=dp_lerobot env=pushT_image dataset_type=lerobot wandb.enable=true training.steps=200000 dataset.keep_video_in_ram=true training.eval_freq=1000 training.log_freq=1000

# Action Chunking with Transformers (ACT) using LeRobot Push-T datasets (pushT_image/aloha_sim_transfer_cube_human)
python scripts/train.py algo=act_lerobot env=aloha_sim_transfer_cube_human dataset_type=lerobot wandb.enable=true training.steps=200000 training.eval_freq=1000 training.log_freq=1000

# Behavior Cloning (BC) using LeRobot Aloha datasets (aloha_sim_transfer_cube_human)
python scripts/train.py algo=bc_lerobot env=aloha_sim_transfer_cube_human dataset_type=lerobot wandb.enable=true training.steps=200000 training.batch_size=16 policy.crop_shape=[440,560] policy.pretrained_backbone_weights=imagenet policy.use_group_norm=false training.eval_freq=20000 training.log_freq=1000

# Diffusion Policy (DP) using LeRobot Aloha datasets (aloha_sim_transfer_cube_human)
python scripts/train.py algo=dp_lerobot env=aloha_sim_transfer_cube_human dataset_type=lerobot wandb.enable=true training.steps=200000 policy.horizon=128 policy.n_action_steps=100 training.batch_size=16 policy.crop_shape=[440,560] policy.pretrained_backbone_weights=imagenet policy.use_group_norm=false training.eval_freq=20000 training.log_freq=1000

```

## License

This project is licensed under the **Apache License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgement

Our implementation of ACT and Diffusion Policy are based on PyTorch versions implemented from [LeRobot](https://github.com/huggingface/lerobot).