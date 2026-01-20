<div align="center">
<h1 align="center">
   If You Can Make an Omelette, Can You Crack an Egg? Probing Zero-Shot Subtask Generalization in Vision-Language-Action Models
</h1>
<p align="center">
  Grigorii Guz, Giuseppe Carenini, Mathias Lécuyer, Michiel van de Panne, Vered Shwartz 
</p>

  [[Paper link]](https://openreview.net/forum?id=h9G9K8gP0q)

</div>

## General installation recommendations

As [Octo](https://github.com/octo-models/octo), [Pi0-FAST](https://github.com/Physical-Intelligence/openpi) and [CogAct](https://github.com/microsoft/CogACT) differ in required dependency versions, we recommend installing each in a separate  ```uv``` environment. 


## Datasets
Download the dataset splits for both CALVIN and LIBERO from [here](https://drive.google.com/drive/folders/1iBrvRj7jTiYCo-ge1Lmy9BRtA8B6N9yu?usp=sharing) and unzip them in at the repository root. The structure should be as follows:
```
repo_name/
  datasets/
    libero_conj
    libero_high_level
    ...
```

## VLAs installation
#### Octo installation
```
cd models/octo
uv venv .venv --python 3.10
source .venv/bin/activate

uv pip install -e .
uv pip install -r requirements.txt
uv pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install --upgrade "nvidia-cudnn-cu11>=8.6,<9"
```

#### Pi0-FAST installation
Follow the instructions from the [original repository](https://github.com/Physical-Intelligence/openpi):
```
cd models/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd ../../
```
In addition, Pi0-FAST requires converting the datasets from the RLDS format (in the link above) to the LeRobot format. 
```
cd models/openpi
uv run examples/libero/convert_calvin_data_to_lerobot.py --data_dir RLDS_DATASETS_PATH
uv run scripts/compute_norm_stats.py --config-name libero_high_level
uv run scripts/compute_norm_stats.py --config-name libero_conj
uv run scripts/compute_norm_stats.py --config-name libero_low_level

uv run scripts/compute_norm_stats.py --config-name calvin_high_level
uv run scripts/compute_norm_stats.py --config-name calvin_conj
uv run scripts/compute_norm_stats.py --config-name calvin_low_level
```

#### CogACT installation

```
cd models/CogACT
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv pip install packaging ninja
uv pip install "flash-attn==2.5.5" --no-build-isolation

cd ../../
cd utils/dlimp
uv pip install -e .
```

## CALVIN and LIBERO installation
The directories `calvin/` and `LIBERO/` contain the modified code for the corresponding [CALVIN](https://github.com/mees/calvin) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmarks with the definitions and evaluation procedures for the high-level and the low-level tasks used in the experiments. Once the environment for a specific VLA is activated, you can install those as follows:

```
cd calvin/calvin_env
uv pip install -r requirements.txt
uv pip install -e .
cd ../../
cd LIBERO
uv pip install -r requirements.txt
uv pip install -e .
```


## Training
Please refer to the ```train_[octo|pi0_fast|cogact].sh``` scripts corresponding to each model. 


## Evaluation
Please refer to the ```eval_[calvin|libero].sh``` scripts corresponding to each environment.  

