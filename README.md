# PyRL - Reinforcement Learning Framework in Pytorch

PyRL is a framework for research in deep reinforcement learning. The following algorithms are implemented in PyTorch:

- [Policy Gradient](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [Deep Q Network](https://www.nature.com/articles/nature14236)
- [DDPG](https://arxiv.org/abs/1509.02971)
- [TD3](https://arxiv.org/abs/1802.09477)
- [PPO](https://arxiv.org/abs/1707.06347)
- [SAC](https://arxiv.org/abs/1801.01290)
- [TRPO](https://arxiv.org/abs/1502.05477) (WIP)
- [ACER](https://arxiv.org/abs/1611.01224) (WIP)
- [ACKTR](https://arxiv.org/abs/1708.05144) (WIP)

This project is still under active development.

# Features

- Modular architecture
- Implemented in PyTorch
- Readable code

# Installation

```
git clone https://github.com/chaovven/pyrl.git
pip3 install -r requirements.txt
```

I highly recommend using conda environment to run the experiments.

Some of the examples use MuJoCo physics simulator. Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py).



# Run Experiments

* Example 1: **TD3**
```
python3 main.py --alg=td3 with env=InvertedPendulum-v2
```

The default arguments are stored in `config/default.yaml`, in which the arguments are shared across all experiments.

The arguments for TD3 are stored in the file `config/algs/td3.yaml`. You can also override the parameters in this file by specifying the arguments in the command-line. For example, if you want to change the value of learning rate (lr), run:
```
python3 main.py --alg=td3 with env=InvertedPendulum-v2 lr=0.0002
```

The same rules are also applicable to other algorithms.


* Example 2: **Policy Gradient**.

```
python3 main.py --alg=pg_box with env=InvertedPendulum-v2 agent=gaussian
```

As policy gradient can be used both in continuous and discrete environment, the suffixes `_box` and `_disc` are used to distinguished between continuous PG and discrete PG. The same naming rule will also be applied to algorithms that can be used in both continuous and discrete environments.

For instance, run policy gradient (`pg_disc`) in `CartPole-v1` environment (discrete):
```
python3 main.py --alg=pg_disc with env=CartPole-v1
```

# Results

The experimental results will be stored in the folder `results` in the form of TensorBoard event file.

Note that the hyperparameters (e.g, learning rate) have not been carefully tuned for each algorithm. 

# Save and Load Models

You can save the model by setting `save_model=True`. Set the model saving frequency by specifying `model_save_interval`. 
For example, if you want to save your model every 10000 timesteps: 
```shell
python3 main.py --alg=pg_disc with env=CartPole-v1 save_mode=True model_save_interval=10000
```
By default, the model will be saved in the directory `./results/{ENV_NAME}/{TENSORBOARD_EVENT_FILE_DIR}/models`.

Load your models by specifying the model path `load_model_path` to be loaded. For example, load your model at timestep 100:
```shell
python3 main.py --alg=pg_disc with env=CartPole-v1 save_mode=True load_model_path=./results/{ENV_NAME}/{TENSORBOARD_EVENT_FILE_DIR}/models/100
```

# Citing PyRL
If you reference or use PyRL in your research, please cite:

```
@article{PyRL2020,
    author = {Wen, Chao},
    title = {{PyRL - Reinforcement Learning Framework in Pytorch}},
    year = {2020}
}
```

This project reuses some code from [pymarl](https://github.com/oxwhirl/pymarl), [spinningup](https://github.com/openai/spinningup.git), and [TD3](https://github.com/sfujim/TD3). Thanks for their work.
