# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:22:02 2025

@author: Elsa_Ehrhart
"""

from environnement import CustomEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import time



# Wrapping the environment for logging
env = CustomEnv()
env = Monitor(env)  # Pour enregistrer les logs de performance

# SAC model
model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0003, gamma=0.99, buffer_size=1000000, batch_size=256, train_freq=1)

# Callback pour sauvegarder le modèle pendant l'entraînement
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/", name_prefix="sac_model")


start=time.time()

# Training the agent
model.learn(total_timesteps=10000, callback=checkpoint_callback)

# Save the final model
model.save("./models/unguided_level_reward")

end=time.time()
