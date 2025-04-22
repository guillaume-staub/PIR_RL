
    
# check_env(CustomEnv(), warn=True, skip_render_check=True)

# stable baseline 3 méthode pour vérifier que l'environnement est ok une fois que tout est prêt
# rechercher stable baseline 3 environment checker : env_checker

# stable baseline algorithme "SAC"
# dans render afficher notre propre métrique d'évaluation
# tirer les séries aléatoirement à chaque reset (aller voir sur RTE pour les données en ligne)

from stable_baselines3 import SAC
from environnement_avecdf import CustomEnv


def eval_model(model_path):
    # Visualiser l'agent entraîné
    model = SAC.load(model_path)
    env = CustomEnv(learning=False)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        env.render()
        

    
        

#for i in range(2,6):
    #train(reward,nb_iter,save_freq,name,config_path="./config.yaml")

#eval_model("./models/guided_step_week_year_10e4")
#eval_model("./models/guided_step_week_year_10e5")
eval_model("./models/guided_step_10e5")
eval_model("./models/guided_year_10e5")
eval_model("./models/guided_week_10e5")
