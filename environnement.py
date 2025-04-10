# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:12:41 2025

@author: Elsa_Ehrhart & Guillaume Staub
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
import yaml
from creation_echantillons import selection_annee_aleatoire
import os

# Loading of the YAML file:
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('config.yaml')
reward_function = config["reward_function"]
update_levels_function = config["update_levels_function"]

print(f"Reward function: {reward_function}")
print(f"Step function: {update_levels_function}")


class CustomEnv(gym.Env):
    def __init__(self,learning=True,test_data=None):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Action meaning:
            #   0: Do nothing with the PHS
            #  -1: Use the PHS as much as we can to provide the demand of electricity
            #   1: Store as much energy as possible in the PHS
        # self.action_space = gym.spaces.Box(low=np.array([min1,min2]), high=np.array([max1,max2]), shape(2,), dtype=np.float32) # if we need more than one action


        # Collection of the data

        demand_data = pd.read_csv('./data/demand2050_ADEME.csv', header=None)
        demand_data.columns = ["time","demand"]
        
        self.learning=learning #on en a besion dans le reset, sur les phse de test on réalisera une série de 2 ans
        
        self.times = demand_data['time'].values
        self.time = 0   # Time indicator
        self.end= self.times[-1]
        self.demand = demand_data['demand'].values
        
        if learning :
            self.region,self.annee,annee_path=selection_annee_aleatoire("./ech_apprentissage")
        elif test_data!=None :
            self.region,self.annee,annee_path=test_data
        else : 
            self.region,self.annee,annee_path=selection_annee_aleatoire("./ech_test")

        solar_path=os.path.join(annee_path,"solar.csv")
        wind_path=os.path.join(annee_path,"wind_onshore.csv")
        self.solar_data = pd.read_csv(solar_path)['facteur_charge'].values
        self.wind_data = pd.read_csv(wind_path)['facteur_charge'].values
        
        #self.solar_data = pd.read_csv('./data/solar.csv')['facteur_charge'].values
        #self.wind_data = pd.read_csv('./data/wind_onshore.csv')['facteur_charge'].values

        #traitement des années bissextiles
        if int(self.annee)%4==0:
            self.nb_jours_annee=366
        else:
            self.nb_jours_annee=365

        # definition of the variables of the environment
       
        self.wind_capacity = 170.1  # given ? or to be calculated ? or to be set with different values to test ?
        self.solar_capacity = 308.4 # max of energy we can gather using wind / sun in one step

        self.phs_capacity = 180
        self.phs_power = 9.3
        self.phs_efficiency = 0.75
        
        self.gas_capacity = 125000
        self.gas_power_in = 7.66
        self.gas_power_out = 32.93
        self.gas_efficiency = 0.4

        # Box observation space with 5 dimensions
        #                                                     time  date   residual prod    phs level   gaz level
        self.observation_space = gym.spaces.Box(low=np.array( [0,   0,    -1,               0,          0]), 
                                                high=np.array([1,   1,     1,               1,          1]),
                                                dtype=np.float64)
        self.state = None
        # Self observation space : State vector (date time PHS levels and so on)
        # Self state : Usually at None, used to initialize values at the very first round
        
        obs, self.eval_data=self.reset()


    def reset(self,seed=None,options=None):
        """Reset the environment to its initial state"""
        if self.learning :
            self.time=np.random.choice(self.times)
            self.end=np.random.choice(self.times)
        else :
            self.time=0
            self.end=0
        self.total_reward = 0
        residual_production=self.wind_capacity*self.wind_data[self.time] + self.solar_capacity*self.solar_data[self.time] - self.demand[self.time]

        # We start at a random day and hour of the year, with the residual production of the year, and the energy tanks half full
        self.state = np.array([self.time%24/24,int(self.time/24)%self.nb_jours_annee/self.nb_jours_annee,residual_production/(self.wind_capacity+self.solar_capacity),1/2,1/2])

        # TO DO : the histogram of the values to see if normalization is pertinent
        obs=self.state

        info={"phs_storage": self.state[3], "gas_storage":self.state[4], "total_furnished_demand" :0,"total_no_furnished_demand":0,"total_reward":0,"nb_heures":(self.end-self.time)%(self.nb_jours_annee*24)}
        #la demande fournie concerne uniquement la demande fournie avec les réserves
        return obs,info
    

    def reward_v1(self,no_furnished_demand):
        if no_furnished_demand>0 :
            reward=-no_furnished_demand/(self.phs_capacity+self.gas_capacity)
        else :
            reward=(self.state[3]*self.phs_capacity+self.gas_capacity*self.state[4])/(self.phs_capacity+self.gas_capacity)
        return reward


    def reward_v2(self,no_furnished_demand):
        if no_furnished_demand>0 :
            reward=-no_furnished_demand
        elif self.state[2]>0 :
            reward=self.state[3]*self.phs_capacity+self.gas_capacity*self.state[4]
        return reward
    
    



###############################################################   U P D A T E   L E V E L   1   #####################################################################################################

    def update_levels_guided(self,action,residual_production):
        phs_in=0 
        gas_in=0 
        phs_usage=0 
        gas_usage=0

        if action[0]>=0 :
            qty_asked_for_phs=self.phs_power*action[0]      # L'action est un pourcentage de la puissance max des PHS
                                                            # Si la production résiduelle est positive, on remplit les PHS autant qu'on peut avec, le reste est fourni par le Gaz
                                                            # Sinon, on fournit autant qu'on peut avec le Gaz pour satisfaire la demande.
                                                            # Si le Gaz suffit, on utilise aussi du Gaz pour remplir les PHS
                                                            # Sinon on vide les PHS pour fournir la demande
           
            if (residual_production > qty_asked_for_phs):
                phs_in = min((1 - self.state[3])*self.phs_capacity, qty_asked_for_phs)
                gas_in = (residual_production - phs_in) #On ajoutera les contraintes à la fin
                
            else:
                gas_usage = min(qty_asked_for_phs - residual_production, self.gas_power_out, self.state[4]*self.gas_efficiency, (1 - self.state[3])*self.phs_capacity-residual_production) #qté de gaz à sortir pour remplir les phs en complément de la prod résiduelle
                    # La quantité de Gaz que l'on va vider pour remplir les PHS et la demande
                
                if (residual_production + gas_usage > 0):
                    phs_in=residual_production+gas_usage #on ajoutera les contraintes à la fin
                    
                else :#ici on le force à vider pour répondre à la demande, j'ai peur que l'agent comprenne pas ce qui se passe => ne rien faire plutot et laisser une demande pas remplie ????
                    phs_usage=min(-(residual_production+gas_usage), self.phs_power, self.state[3]*self.phs_efficiency)
                    
        else: # on vide les phs
            phs_usage=min(-self.phs_power*action[0], self.state[3]*self.phs_efficiency*self.phs_capacity)

            if residual_production+phs_usage>0 : #on rempli le gaz
                gas_in=phs_usage+residual_production

            else : #on vide aussi du gaz pour essayer de répondre à la demande
                gas_usage=min(self.state[4]*self.gas_capacity*self.gas_efficiency,self.gas_power_out,-(residual_production+phs_usage))
            
        # Adding the Constraints
        phs_in=min(phs_in,self.phs_power,self.phs_capacity-self.state[3]*self.phs_capacity)
        gas_in=min(gas_in,self.gas_power_in,self.gas_capacity-self.state[4]*self.gas_capacity)
        
        # Update of the capacity levels
        state3=self.state[3]+phs_in/self.phs_capacity-phs_usage/self.phs_efficiency/self.phs_capacity
        state4=self.state[4]+gas_in/self.gas_capacity-gas_usage/self.gas_efficiency/self.gas_capacity
        
        # Computation of the unfurnished demand
        no_furnished_demand=max(0,-residual_production-gas_usage-phs_usage)
        
        # For unit tests
        return (state3,state4,no_furnished_demand, phs_in, gas_in, phs_usage, gas_usage)
    

###############################################################   U P D A T E   L E V E L   2   #####################################################################################################
    
    # We do as the agent says, only physical limits can override its action
    def update_levels_unguided(self,action,residual_production):
        phs_in=0 
        gas_in=0 
        phs_usage=0 
        gas_usage=0

        if action[0]>=0 :
            qty_asked_for_phs=self.phs_power*action[0] 
             
            if (residual_production > qty_asked_for_phs):
                phs_in = min((1 - self.state[3])*self.phs_capacity, qty_asked_for_phs)
                gas_in = (residual_production - qty_asked_for_phs) #perte d'énergie si l'agent veut trop remplir les phs
                     
            else:
                gas_usage = min(qty_asked_for_phs - residual_production, self.gas_power_out, self.state[4]*self.gas_efficiency)#, (1 - self.state[3])*self.phs_capacity-residual_production) #qté de gaz à sortir pour remplir les phs en complément de la prod résiduelle
                # La quantité de Gaz que l'on va vider pour remplir les PHS et la demande
                phs_in=min(qty_asked_for_phs, gas_usage, (1 - self.state[3])*self.phs_capacity) #on ajoutera les contraintes à la fin
                no_furnished_demand=max(0,gas_usage-phs_in+residual_production)
         
                
        else : # on vide les phs
            phs_usage=min(-self.phs_power*action[0], self.state[3]*self.phs_efficiency*self.phs_capacity)
            if residual_production+phs_usage>0 : #on rempli le gaz
                gas_in=phs_usage+residual_production
            else : #on vide aussi du gaz pour essayer de répondre à la demande
                gas_usage=min(self.state[4]*self.gas_capacity*self.gas_efficiency,self.gas_power_out,-(residual_production+phs_usage))
         
        #ajout des contraintes
        phs_in=min(phs_in,self.phs_power,self.phs_capacity-self.state[3]*self.phs_capacity)
        gas_in=min(gas_in,self.gas_power_in,self.gas_capacity-self.state[4]*self.gas_capacity)
         
        #mise à jour des etats
        state3=self.state[3]+phs_in/self.phs_capacity-phs_usage/self.phs_efficiency/self.phs_capacity
        state4=self.state[4]+gas_in/self.gas_capacity-gas_usage/self.gas_efficiency/self.gas_capacity
         
        #calcul demande non fournie
        no_furnished_demand=max(0,-residual_production-gas_usage-phs_usage)
                                                       #pour des tests unitaires
        return (state3,state4,no_furnished_demand, phs_in, gas_in, phs_usage, gas_usage)


    reward_functions = {
        "reward_v1": reward_v1,
        "reward_v2": reward_v2
    }
    
    update_levels_functions = {
    "update_levels_guided": update_levels_guided,
    "update_levels_unguided": update_levels_unguided
    }

    reward_function = reward_functions[reward_function]
    update_level_function = update_levels_functions[update_levels_function]


    def step(self, action):
        """Take an action and return the new state, reward, done, and info"""
        assert self.action_space.contains(action), "Invalid action!"


        # Add later year periodicity
        # Time update at each step, The first value corresponds to the number of days
        # (a float, the decimal part representing the progress of the current day).
        # The second one is a 24hours periodic function, so the agent can better understand how days work,
        # understanding for example that every night the consumption decreases
      
        self.time = (self.time + 1) % len(self.times)
        self.state[0]=self.time%24/24 # heure
        self.state[1]=(int(self.time/24)%self.nb_jours_annee)/self.nb_jours_annee #jour
        
        
        residual_production = self.wind_capacity*self.wind_data[self.time] + self.solar_capacity*self.solar_data[self.time] - self.demand[self.time]
      
        self.state[2] = residual_production/(self.wind_capacity+self.solar_capacity) #taux par rapport à la production maximale possible, majoré par 1 mais peut être inférieure à -1 si la demande est deux fois plus importante que l'offre
        
        self.state[3],self.state[4],no_furnished_demand,phs_in, gas_in, phs_usage, gas_usage=self.update_level_function(action,residual_production)
        
        
        
        # Tests unitaires pour vérifier cohérence des résultats

        #tests sur les bornes des états
        assert (self.state[0]>=0 or self.state[0] <= 1), "L'état 0 est hors des limites définies !"
        assert (self.state[1]>=0 or self.state[1] <= 1), "L'état 1 est hors des limites définies !"
        assert (self.state[2]<=1), "L'état 2 est hors des limites définies !"
        assert (self.state[3]>=0 or self.state[3] <= 1), "L'état 3 est hors des limites définies !"
        assert (self.state[4]>=0 or self.state[4] <= 1), "L'état 4 est hors des limites définies !"
        
        #test à rajouter pour vérifier que si l'action est de vider
        assert (action >= 0 or (action[0] < 0 and phs_in <= 0)), "L'agent demande de vider les phs et ils se remplissent"
        #test pour vérifier qu'on ne créé pas d'énergie
        assert ((residual_production <= 0) or (residual_production > 0 and round(residual_production, 6) >= round(phs_in+gas_in-phs_usage-gas_usage, 6))), "On a créé de l'énergie, vite déposez un brevet !!!"
        
        #vérification des débits
        assert (phs_in<=self.phs_power), "Le remplissage des phs est suppérieur au débit possible"
        assert (phs_usage<=self.phs_power), "L'utilisation des phs est suppérieure au débit possible"
        assert (gas_in<=self.gas_power_in), "Le remplissage du gaz est suppérieur au débit possible"
        assert (gas_usage<=self.gas_power_out), "L'utilisation des phs est suppérieure au débit possible"
      
        # Reward function
        # cout basé sur la pénalité d'écrétage / de perte par conversion d'énergie ?
        
        reward = self.reward_function(no_furnished_demand)

        # info : evaluation data
        
        self.eval_data["phs_storage"]=self.state[3]
        self.eval_data["gas_storage"]=self.state[4]
        self.eval_data["total_reward"]+=reward
        if residual_production<0 :
            self.eval_data["total_furnished_demand"]-=(residual_production+no_furnished_demand)
            self.eval_data["total_no_furnished_demand"]+=no_furnished_demand
        

        
        

        

        # Termination condition
        terminated = bool(self.time==self.end) 
        
        truncated=False

        return self.state, reward, terminated, truncated, self.eval_data[-1]


    def render(self, mode='human'):
        """Render the environment (optional)"""
        if self.time==self.end :
            print(self.eval_data)
            

    def close(self):
        """Clean up resources (optional)"""

check_env(CustomEnv(), warn=True, skip_render_check=True)   

#1 an cyclique, début n'importe quand dans l'année, on ne dure pas toujours 1 an, qqté non fournie

#critère de réussite : combien il a fourni, combien il a stocké
#évaluation sur une quarantaine de séries
#regarder si stable baseline peut évaluer sur un échantillon d'entrainement
#retourner un dict avec les indicateurs qui nous intéressent
#création config