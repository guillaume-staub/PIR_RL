# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:22:02 2025

@author: Elsa_Ehrhart
"""

from environnement_avecdf import CustomEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import yaml

def train(reward,nb_iter,save_freq,name,config_path="./config.yaml") :
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['reward_function'] = reward
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    
    # Wrapping the environment for logging
    env = CustomEnv()
    env = Monitor(env)  # Pour enregistrer les logs de performance
    # SAC model
    model = SAC("MlpPolicy", env, verbose=1)#, learning_rate=0.0003, gamma=0.99, buffer_size=1000000, batch_size=256, train_freq=1)
    # Callback pour sauvegarder le modèle pendant l'entraînement
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./models/", name_prefix="tmp_save")
    start=time.time()
    # Training the agent
    model.learn(total_timesteps=nb_iter, callback=checkpoint_callback)
    # Save the final model
    model.save("./models/"+name)
    end=time.time()
    
    print("name")
    print("time elapsed")
    print(end-start)

train("reward_v1",10000,10000,"test")
#2years_guided_reward_weekly_and_hours_3_10e5