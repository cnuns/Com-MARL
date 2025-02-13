# Communication-Aware Graph Neural Network for Multi-Agent Reinforcement Learning

## Requirements
- PyTorch
- ma-gym

## Installation
It is recommended to run the code in a [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) virtual environment.

1. Create a virtual environment:
    ```sh
    conda create -n commarl python=3.7
    ```
2. Activate the virtual environment:
    ```sh
    conda activate commarl
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Model Configuration
The core framework files are located in `/com_marl/torch/`. The neural network architectures used in the framework can be found in the following subdirectories:

- `/com_marl/torch/baselines/`
- `/com_marl/torch/modules/`
- `/com_marl/torch/policies/`

### Approaches & File Locations
- `sn`: scenario name (`pp` for Predator-Prey or `co` for Coverage)

| Approach | Runner File | Policy Network (`policies/`) | Value Network (`baselines/`) |
|----------|------------|-----------------------------|-----------------------------|
| CENT | `runner_{sn}_cent.py` | `centralized_categorical_mlp_policy.py` | `gaussian_mlp_baseline.py` |
| Comm-DP | `runner_{sn}_commDP.py` | `comm_categorical_mlp_policy.py` | `comm_base_critic.py` |
| Obs-DP | `runner_{sn}_obsDP.py` | `dec_categorical_mlp_policy.py` | `comm_base_critic.py` |

## Experiments

### Predator-Prey
The experiment runner scripts for the Predator-Prey environment are located in `/exp_runners/predatorprey/`. Each script includes detailed argument specifications, such as environment size, number of agents, network configurations, and algorithm hyperparameters.

To run training in the Predator-Prey environment, navigate to the corresponding directory:
```sh
cd /exp_runners/predatorprey/
```
Then, execute the following command, replacing the placeholders with your desired values:

- `M`: Map size (e.g., 10, 20, 30, or multiples of 10)
- `S`: Sensing range (1 or 2)
- `D`: Node density (0.04 or 0.08)
- `C`: Capture requirement (2 or 4)
- `P`: Packet loss probability, where `P ∈ [0, 1]`, default is 0.

| Approach | Command |
|----------|---------|
| CENT | `python runner_pp_cent.py --cmd (train or test) --map M --sen S --den D --cap C --loss P` |
| Comm-DP | `python runner_pp_comm.py --cmd (train or test) --map M --sen S --den D --cap C --loss P` |
| Obs-DP | `python runner_pp_obs.py --cmd (train or test) --map M --sen S --den D --cap C --loss P` |

Example:
```sh
python runner_pp_comm.py --cmd train --map 10 --sen 1 --den 0.04 --cap 2 --loss 0
```

### Coverage
The experiment runner scripts for the Coverage environment are located in `/exp_runners/coverage/`. Each script includes detailed argument specifications similar to those in the Predator-Prey environment.

To run training in the Coverage environment, navigate to the corresponding directory:
```sh
cd /exp_runners/coverage/
```
Then, execute the following command, replacing the placeholders with your desired values:

- `M`: Map size (e.g., 10, 20, 30, or multiples of 10)
- `S`: Sensing range (1 or 2)
- `D`: Node density (0.03 or 0.06)
- `P`: Packet loss probability, where `P ∈ [0, 1]`, default is 0.

| Approach | Command |
|--|--|
| CENT |`python runner_co_cent.py --cmd (train or test) --map M --sen S --den D --loss P`|
| Comm-DP | `python runner_co_comm.py --cmd (train or test) --map M --sen S --den D --loss P` |
| Obs-DP | `python runner_co_obs.py --cmd (train or test) --map M --sen S --den D --loss P` |

Example,
```sh
python runner_co_comm.py --cmd train --map 10 --sen 1 --den 0.03 --loss 0 
```
### Training Checkpoints & Logs
- Model checkpoints are stored in `/exp_runners/{scenario_name}/data/model/`.
- Training logs and results are saved in `/exp_runners/{scenario_name}/data/model/{setup_name}/`.

Files generated for each experiment setup:
- `itrs/`: Policy network checkpoints for each epoch.
- `debug.log`: Training log file.
- `params.pkl`: Latest training state, including policy and value networks, environment details, and metadata.
- `progress.csv`: Log file containing total environment steps, epoch iterations, average rewards, etc.
- `variant.json`: Hyperparameter settings for the specific experiment setup.

### Testing Checkpoints & Logs
- Test results are stored in `/exp_runners/{scenario_name}/data/test/`.
- `./test/param/`: Directory containing test parameter setups.
- `./test/csv/`: Directory where test results are saved in `.csv` format.
- `./test/matlab/`: Directory where test results are saved in MATLAB `.mat` format.
