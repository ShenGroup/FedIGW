import pandas as pd
import numpy as np
import time
from utils.utils import parse_data
from sklearn.linear_model import LogisticRegression
from contextualbandits.online import AdaptiveGreedy
from copy import deepcopy


def simulate_rounds(model, rewards, actions_hist, X_global, y_global, batch_st, batch_end):
    np.random.seed(int(time.time()))
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(batch_st)
    model.fit(X_global[:batch_end, :], new_actions_hist, y_global[np.arange(batch_end), new_actions_hist])
    print(f"Current round {batch_st // 50}, reward: {rewards[-1] / (batch_end - batch_st): .2f}")
    
    return new_actions_hist


def get_mean_reward(reward_lst, batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew

def run(num_rounds, dataset):
    '''
    rounds:int, each round with go over the entire dataset once.
    '''
    # breakpoint()
    X, y = parse_data(f"./datasets/{dataset}.txt")
    nchoices = y.shape[1]
    base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
    beta_prior = ((3./nchoices, 4), 2)

    adaptive_greedy_thr = AdaptiveGreedy(deepcopy(base_algorithm), nchoices=nchoices,
                                     decay_type='threshold',
                                     beta_prior = beta_prior, random_state = 6666)

    rewards_agr = list()
    batch_size = 50
    first_batch = X[:batch_size, :]
    np.random.seed(int(time.time()))
    action_chosen = np.random.randint(nchoices, size=batch_size)
    rewards_received = y[np.arange(batch_size), action_chosen]
    lst_a_agr = action_chosen.copy()

    adaptive_greedy_thr.fit(X = first_batch, a = action_chosen, r = rewards_received)
    time_n = int(time.time())
    
    for _ in range(num_rounds):
        lst_a_agr = action_chosen.copy()
        for i in range(int(np.floor(X.shape[0] / batch_size))):
            batch_st = (i + 1) * batch_size
            batch_end = (i + 2) * batch_size
            batch_end = np.min([batch_end, X.shape[0]])
            lst_a_agr = simulate_rounds(adaptive_greedy_thr,
                                        rewards_agr,
                                        lst_a_agr,
                                        X,y,
                                        batch_st, batch_end)
            
    rounds = 50 * np.arange(len(rewards_agr))
    # breakpoint()
    df_log = pd.DataFrame({
        "rounds":rounds,
        "reward":get_mean_reward(rewards_agr,batch_size=batch_size)
    })
    df_log.to_csv(f"../results/{dataset}_method_agr_{num_rounds}_time_{time_n}.csv", index = False)
    
    


