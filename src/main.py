from fl.fl_server import FL_Server
from baselines.context_baseline import run
import argparse



def main():

    for _ in range(9):
        # server2 = FL_Server("Bibtex", "MLP", 1, 7000, "FedAvg")
        # server2.train(1, 0 , 100)
        # server = FL_Server("Bibtex", "MLP", 10, 7000, "FedAvg")
        # server.train(1, 0 , 100, 10)
        
        # server2 = FL_Server(env_name = "Bibtex",  reward_function = "MLP", client_num = 10, gamma = 7000, fl_method = "FedAvg", sample_method="default")
        # server2.train(comm_round = 1, local_steps = 0 , total_rounds = 100, client_num = 10)
        # server2 = FL_Server(env_name = "Bibtex",  reward_function = "MLP", client_num = 10, gamma = 7000, fl_method = "FedAvg", sample_method="greedy")
        # server2.train(comm_round = 1, local_steps = 0 , total_rounds = 100, client_num = 10)
        # server2 = FL_Server("Bibtex", "MLP", 10, 7000, "Scaffold")
        # server2.train(1, 0 , 100, 10)
        # server = FL_Server("Bibtex", "MLP", 50, 7000, "FedProx")
        # server.train(1, 0 , 100, 50)
        # server = FL_Server("Bibtex", "MLP", 50, 7000, "Scaffold")
        # server.train(1, 0 , 100, 50)
    #     server2 = FL_Server("Bibtex", "MLP", 20, 7000, "FedAvg")
    #     server2.train(1, 0 , 100)
    #     server = FL_Server("Bibtex", "MLP", 30, 7000, "FedAvg")
    #     server.train(1, 0 , 100)
        # server2 = FL_Server("Delicious", "MLP", 10, 7000, "FedAvg")
        # server2.train(1, 0 , 100, 10)
        # server = FL_Server(env_name = "Delicious",  reward_function = "MLP", client_num = 10, gamma = 7000, fl_method = "FedAvg", sample_method="greedy")
        # server.train(1, 0 , 100, 10)
        server = FL_Server(env_name = "Delicious",  reward_function = "MLP", client_num = 10, gamma = 10000, fl_method = "FedAvg", sample_method="default")
        server.train(1, 0 , 100, 10)
        # server = FL_Server("Delicious", "MLP", 10, 7000, "Scaffold")
        # server.train(1, 0 , 100, 10)
        # server = FL_Server("Delicious", "MLP", 50, 7000, "FedProx")
        # server.train(1, 0 , 100, 50)
        # server = FL_Server("Delicious", "MLP", 50, 7000, "Scaffold")
        # server.train(1, 0 , 100, 50)
        # server2 = FL_Server("Delicious", "MLP", 50, 7000, "FedAvg")
        # server2.train(1, 0 , 100, 50)
    #     server2 = FL_Server("Delicious", "MLP", 20, 7000, "FedAvg")
    #     server2.train(1, 0 , 100)
    #     server = FL_Server("Delicious", "MLP", 30, 7000, "FedAvg")
    #     server.train(1, 0 , 100)
        # run(3, "Bibtex")
        # run(2, "Delicious")
    
if __name__ == "__main__":
    main()