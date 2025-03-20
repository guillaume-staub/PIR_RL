
    
# check_env(CustomEnv(), warn=True, skip_render_check=True)

# stable baseline 3 méthode pour vérifier que l'environnement est ok une fois que tout est prêt
# rechercher stable baseline 3 environment checker : env_checker

# stable baseline algorithme "SAC"
# dans render afficher notre propre métrique d'évaluation
# tirer les séries aléatoirement à chaque reset (aller voir sur RTE pour les données en ligne)

from stable_baselines3 import SAC
from environnement import CustomEnv




# Visualiser l'agent entraîné
model = SAC.load("./models/guided_level_reward")
env = CustomEnv()

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()