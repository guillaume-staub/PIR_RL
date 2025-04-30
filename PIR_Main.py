
    
# check_env(CustomEnv(), warn=True, skip_render_check=True)

# stable baseline 3 méthode pour vérifier que l'environnement est ok une fois que tout est prêt
# rechercher stable baseline 3 environment checker : env_checker

# stable baseline algorithme "SAC"
# dans render afficher notre propre métrique d'évaluation
# tirer les séries aléatoirement à chaque reset (aller voir sur RTE pour les données en ligne)

from stable_baselines3 import SAC
from environnement_avecdf import CustomEnv
import yaml

def eval_model_train(model_path,path="") :
        

    # Visualiser l'agent entraîné
    model = SAC.load(model_path)
    env = CustomEnv(learning=False,annee1_path=path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        
    return info
        

def eval_model(model_path,reward,update_levels,config_path="./config.yaml",path="") :
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['reward_function'] = reward
    config['update_levels_function'] = update_levels
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)
    
    # Visualiser l'agent entraîné
    model = SAC.load(model_path)
    env = CustomEnv(learning=False,annee1_path=path)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        
    return info

    
        

#for i in range(2,6):
    #train(reward,nb_iter,save_freq,name,config_path="./config.yaml")

#eval_model("./models/guided_step_week_year_10e4")
#eval_model("./models/guided_step_week_year_10e5")
path="./data"
#♥eval_model("./models/unguided_v5_1_double_power","reward_v5","update_levels_unguided",path=path)
#eval_model("./models/guided_year_10e5")
#eval_model("./models/guided_week_10e5")
