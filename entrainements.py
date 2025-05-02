# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:22:02 2025

@author: Elsa_Ehrhart
"""

from environnement_avecdf import CustomEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from PIR_Main import eval_model_train
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
    env = Monitor(env)  # Pour enregistrer les logs de performance
    # SAC model
    model = SAC("MlpPolicy", env, verbose=1)#, learning_rate=0.0003, gamma=0.99, buffer_size=1000000, batch_size=256, train_freq=1)
    # Callback pour sauvegarder le modèle pendant l'entraînement
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./models/", name_prefix="tmp_save")
    
    
    
    start=time.time()
    # Training the agent
    
    n=1000
    nb_eval=nb_iter//n
    path_eval="./data"
    
    reward=np.zeros(nb_eval)
    
    for i in range (nb_eval) : 
        model.learn(total_timesteps=n, callback=checkpoint_callback)
        model_path="./models/"+name
        model.save(model_path)
        info=eval_model_train(model_path,path=path_eval)
        reward[i]=info["total_reward"]
        
    # Save the final model
    model.save("./models/"+name)
    end=time.time()
    
    print("name")
    print("time elapsed")
    print(end-start)
    plt.title("Reward evolution every "+str(n)+" trainings")
    plt.plot(reward)

#train("reward_v1","update_levels_guided",1000,1000,"guided_step_week_year_10e3")

path="./data"

train("reward_v6","update_levels_guided",100000,100000,"guided_v6_10e5_overfitting_double_power",path=path)

train("reward_v7","update_levels_guided",100000,100000,"guided_v7_10e5_double_power")

train("reward_v7","update_levels_unguided",100000,100000,"unguided_v7_10e5_overfitting_double_power",path=path)
train("reward_v6","update_levels_unguided",100000,100000,"unguided_v6_10e5_overfitting_double_power",path=path)

train("reward_v7","update_levels_unguided",100000,100000,"unguided_v7_10e5_double_power")


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