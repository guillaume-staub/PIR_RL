# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:22:02 2025

@author: Elsa_Ehrhart
"""

from environnement_avecdf import CustomEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from PIR_Main import eval_model_train
from PIR_Main import eval_model
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

def train(reward,update_levels,nb_iter,save_freq,name,config_path="./config.yaml",path="") :
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['reward_function'] = reward
    config['update_levels_function'] = update_levels
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    
    # Wrapping the environment for logging
    env = CustomEnv(annee1_path=path)
    # SAC model
    model = SAC("MlpPolicy", env, verbose=1,buffer_size=1000000,gamma=0.999)#, learning_rate=0.0003, gamma=0.99, buffer_size=1000000, batch_size=256, train_freq=1)
    # Callback pour sauvegarder le modèle pendant l'entraînement
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./models/")
    
    
    
    start=time.time()
    # Training the agent
    
    n=save_freq
    nb_eval=nb_iter//n
    path_eval="./data"
    
    reward=np.zeros(nb_eval)
    model_path="./models/"+name
    
    for i in range (nb_eval) : 
        model.learn(total_timesteps=n, callback=checkpoint_callback)
        
        model.save(model_path)
        info=eval_model_train(model_path,path=path_eval)
        reward[i]=info["total_reward"]
        
    # Save the final model
    model.save(model_path)
    end=time.time()
    
    print("name")
    print("time elapsed")
    print(end-start)
    plt.title("Reward evolution every "+str(n)+" trainings")
    plt.plot(reward)
    plt.show()
    
    
def train_evalcallback(reward,update_levels,nb_iter,save_freq,name,config_path="./config.yaml",path="") :
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['reward_function'] = reward
    config['update_levels_function'] = update_levels
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    
    # Wrapping the environment for logging
    env = CustomEnv(annee1_path=path)
    env_eval = CustomEnv(learning=False,annee1_path="./data")
    # SAC model
    model = SAC("MlpPolicy", env, verbose=1,buffer_size=1000000,gamma=0.999,learning_rate=0.00003,batch_size=512)#, learning_rate=0.0003, gamma=0.99, buffer_size=1000000, batch_size=256, train_freq=1)
    
    
        
    n=save_freq
    nb_eval=nb_iter//n
    path_eval="./data"
    
    reward=np.zeros(nb_eval)
    model_path="./models/"+name
    
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./models/", name_prefix="tmp_save")
    eval_callback = EvalCallback(env_eval, best_model_save_path='./models/'+name,
                                 log_path='./logs/'+name, eval_freq=n,
                                 deterministic=True, render=False)
    
    start=time.time()
    
    model.learn(total_timesteps=nb_iter, callback=[checkpoint_callback, eval_callback])
    
    # Save the final model
    model.save(model_path)
    end=time.time()
    
    print(name)
    print("time elapsed")
    print(end-start)
    
    

def plot_eval_rewards_from_file(file_path):
    data = np.load(file_path)
    
    timesteps = data["timesteps"]            # ex: [10000, 20000, 30000, ...]
    results = data["results"]                # ex: array([[r1, r2, r3], [r1, r2, r3], ...])
    
    rewards = results.mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards)
    plt.xlabel("Number of trainings")
    plt.ylabel("Reward")
    plt.title("Evolution of the reward while training")
    plt.tight_layout()
    plt.show()


#train("reward_v1","update_levels_guided",1000,1000,"guided_step_week_year_10e3")

path="./data"
#train_evalcallback("reward_v12","update_levels_unguided",100000,100,"v12_unguided_100000_100_double_power")
#train_evalcallback("reward_v4","update_levels_unguided",1000,100,"v4_unguided_1000_100")

name="./logs/v12_unguided_100000_100_overfitted_double_power/evaluations.npz"
plot_eval_rewards_from_file(name)

eval_model("./models/v12_unguided_100000_100_overfitted_double_power/best_model.zip","reward_v1","update_levels_unguided",path=path)
#plot_eval_rewards_from_file("./logs/test_unguided_v1_100_10_cb/evaluations.npz")
#train("reward_v1","update_levels_unguided",100,10,"test_unguided_v1_10")



#train_evalcallback("reward_v1","update_levels_guided",400000,1000,"guided_v1_4-10e5_double_power")

#train_evalcallback("reward_v1","update_levels_unguided",400000,1000,"unguided_v1_4-10e5_overfitting_double_power",path=path)

#train_evalcallback("reward_v1","update_levels_unguided",400000,1000,"unguided_v1_4-10e5_double_power")

#train("reward_v7","update_levels_guided",100000,100000,"guided_v7_10e5_double_power")

#train("reward_v7","update_levels_unguided",100000,100000,"unguided_v7_10e5_overfitting_double_power",path=path)
#train("reward_v6","update_levels_unguided",100000,100000,"unguided_v6_10e5_overfitting_double_power",path=path)

#train("reward_v7","update_levels_unguided",100000,100000,"unguided_v7_10e5_double_power")


#train("reward_v5","update_levels_unguided",100000,100000,"unguided_v5_10e5")

#train("reward_v6","update_levels_unguided",100000,100000,"unguided_v6_10e5_overfitting",path=path)

#train("reward_v6","update_levels_unguided",100000,100000,"unguided_v6_10e5_double_power")

#train("reward_v6","update_levels_guided",100000,100000,"guided_v6_10e5_double_power")

#train("reward_v6","update_levels_guided",100000,100000,"guided_v5_10e5_double_power")


#train("reward_v5","update_levels_guided",100000,100000,"guided_v5_10e5_overfitting",path=path)

#train("reward_v5","update_levels_guided",100000,100000,"guided_v5_10e5")

#train("reward_v6","update_levels_guided",100000,100000,"guided_v6_10e5_overfitting",path=path)

#train("reward_v6","update_levels_guided",100000,100000,"guided_v6_10e5")


#train("reward_v3","update_levels_guided",100000,100000,"guided_year_10e5")
#train("reward_v4","update_levels_guided",100000,1000000,"guided_week_10e5")

#train("reward_v2","update_levels_unguided",100000,100000,"unguided_step_10e5")
#train("reward_v3","update_levels_unguided",100000,100000,"unguided_year_10e5")
#train("reward_v4","update_levels_unguided",100000,1000000,"unguided_week_10e5")

#train("reward_v1",1000000,1000000,"guided_step_week_year_10e6")


#2years_guided_reward_weekly_and_hours_3_10e5

#possibilité évaluation en parallèle

