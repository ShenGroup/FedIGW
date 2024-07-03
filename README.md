# FedIGW

This repository is the official implementation of the algorithm FedIGW for the paper [Harnessing the Power of Federated Learning in Federated Contextual Bandits](https://arxiv.org/abs/2312.16341), which will be published in TMLR soon. 


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

For the baseline AGR, we adopted some implementation from the github repo [contextualbandits](https://github.com/david-cortes/contextualbandits/tree/master) and the slightly revised code is placed in /src/baseline folder, which can be directly used.

For the baseline FN-UCB, we directly utilized the code from the authors of [Federated Neural Bandits](https://github.com/daizhongxiang/Federated-Neural-Bandits/tree/main) please clone the code from repo and substitute some of the training scripts use the code in ./src/FNB. To run the experiment, follow the instruction of that repo.

Before running the script, you need to change the path config in env.py:

```env_set
dataset_path = "/path/to/your/dataset"
```

## Training

To train the model(s) in the paper, run this command:

```train
python ./src/main.py
```

For the server definition, you can control whether using 'Bibtex' or 'Delicious'. For reward function, we support 'MLP' and 'CNN'. Client number here stands for the total number of clients, we support 'FedAvg',
'FedProx' and 'Scaffold' as our fl structure right now.

```server
FL_Server(env_name = "Bibtex",  reward_function = "MLP", client_num = 10, gamma = 7000, fl_method = "FedProx")
```

For the training steps: comm_rounds stands for how often the clients will sync with the server. The local step stands for how many local training steps the client will take for each local training round, 0 will follow a default exponential growing strategy with largest step of 4096. Client number here stands for how many client will join the communication each communication round.

```training
train(comm_round = 1, local_steps = 0 , total_rounds = 100, client_num = 10)
```


## Results

After running the training script, you can then plot the figures similar to our paper following the jupyter notebook, notice that before drawing to results, you need to specify the path you where you log your results:


```env_set
result_path = "/path/to/your/result"
```


