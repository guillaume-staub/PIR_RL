# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 13:12:41 2025

@author: Elsa_Ehrhart & Guillaume Staub
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env


    #### renewable ninja

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        # Example: Continuous action space with 1 dimension
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        #0 ne pas toucher au PHS, -1 se vider de la plus haute valeur possible, 1 stocker de la plus haute valeur possible
        # self.action_space = gym.spaces.Box(low=np.array([min1,min2]), high=np.array([max1,max2]), shape(2,), dtype=np.float32) #si besoin de plus que une action

        # Collection of the data
        demand_data=pd.read_csv('./data/demand2050_ADEME.csv', header=None)
        demand_data.columns = ["time","demand"]
        self.times=demand_data['time'].values
        self.time=0 #curseur de temps
        self.demand=demand_data['demand'].values
        self.solar_data=pd.read_csv('./data/solar.csv')['facteur_charge'].values
        self.wind_data=pd.read_csv('./data/wind_onshore.csv')['facteur_charge'].values

        # definition of the variables of the environment


       
        self.wind_capacity = 170.1     # given ? or to be calculated ? or to be set with different values to test ?
        self.solar_capacity = 308.4    # max of energy we can gather using wind / sun in one step

        self.phs_capacity=180
        self.phs_power=9.3
        self.phs_efficiency=0.75
        
        self.gas_capacity=125000
        self.gas_power_in=7.66
        self.gas_power_out=32.93
        self.gas_efficiency=0.4

        # Box observation space with 5 dimensions
        #                                                     time  date   residual prod    phs level   gaz level
        self.observation_space = gym.spaces.Box(low=np.array( [0,   0,    -1,               0,          0]), 
                                                high=np.array([1,   1,     1,               1,          1]),
                                                dtype=np.float64)

        self.state = None
        # Self observation space : le vecteur des états (date heure niveau phs etc)
        # Self state : pour initialiser dès le premier tour les variables, habituellement mis a None
        info,obs=self.reset()
      
    def reset(self,seed=None,options=None):
        """Reset the environment to its initial state"""
        self.time=0 # np.random.choice(self.times)
        self.total_reward = 0
        residual_production=self.wind_capacity*self.wind_data[self.time] + self.solar_capacity*self.solar_data[self.time] - self.demand[self.time]
        
        #on commence à une heure et un jour aléatoire de l'année, avec la production résiduel de ce jour et les réservoirs à moitié pleins
        self.state = np.array([self.time%24/24,int(self.time/24)%365/365,residual_production/(self.wind_capacity+self.solar_capacity),1/2,1/2]) # Voir hypothèses (stock au milieu de la capa max)
        # faire l'histogramme des valeurs, voir si la normalisation est pertinente
        obs=self.state
        info=dict()
        return obs,info
    
    
    
    def reward(self,no_furnished_demand):
        if no_furnished_demand>0 :
            reward=-1
        else :
            reward=1
        return reward



    def level_reward(self,no_furnished_demand):
        if no_furnished_demand>0 :
            reward=-1
        else :
            reward=1+self.state[3]+self.state[4]
        return reward  
    
    def level_and_gas_reward(self,no_furnished_demand):
        if no_furnished_demand>0 :
            reward=-1
        else :
            reward=10+self.state[3]+self.state[4]*1.2
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
    

###############################################################   U P D A T E   L E V E L   2   #####################################################################################################
    def update_levels_unguided(self,action,residual_production):  #on essaye de faire ce que veut l'agent, seules les limites physiques le contraignent

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


    def step(self, action):
        """Take an action and return the new state, reward, done, and info"""
        assert self.action_space.contains(action), "Invalid action!"


        # Add later year periodicity
        # Time update at each step, The first value corresponds to the number of days
        # (a float, the decimal part representing the progress of the current day).
        # The second one is a 24hours periodic function, so the agent can better understand how days work,
        # understanding for example that every night the consumption decreases
      
        self.time += 1
        self.state[0]=self.time%24/24 # heure
        self.state[1]=(int(self.time/24)%365)/365 #jour
        
        
        residual_production = self.wind_capacity*self.wind_data[self.time] + self.solar_capacity*self.solar_data[self.time] - self.demand[self.time]
      
        self.state[2] = residual_production/(self.wind_capacity+self.solar_capacity) #taux par rapport à la production maximale possible, majoré par 1 mais peut être inférieure à -1 si la demande est deux fois plus importante que l'offre
        
        self.state[3],self.state[4],no_furnished_demand,phs_in, gas_in, phs_usage, gas_usage=self.update_levels_unguided(action,residual_production)

        # Tests unitaires pour vérifier cohérence des résultats

        #tests sur les bornes des états
        assert (self.state[0]>=0 or self.state[0] <= 1), "L'état 0 est hors des limites définies !"
        assert (self.state[1]>=0 or self.state[1] <= 1), "L'état 1 est hors des limites définies !"
        assert (self.state[2] <= 1), "L'état 2 est hors des limites définies !"
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
      
        reward=self.level_reward(no_furnished_demand)
        self.total_reward += reward

        # Example: Termination condition
        terminated = bool(self.state[1]==int(self.times[-1]/24)%365/365 and self.state[0]==self.times[-1]%24/24) #finit à la fin de la série
        # done=self.times[-1]==self.time
        info = {}  # Additional information (can be empty)
        
        truncated=False

        return self.state, reward, terminated, truncated, info


    def render(self, reward, mode='human'):
        """Render the environment (optional)"""
        if self.time==self.times[-1] :
            print(f"State: {self.state}")
            print(f"Total reward: {self.total_reward}")

    def close(self):
        """Clean up resources (optional)"""

check_env(CustomEnv(), warn=True, skip_render_check=True)   
