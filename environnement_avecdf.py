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
import matplotlib.pyplot as plt

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
    def __init__(self,learning=True,annee1_path=""):
        super(CustomEnv, self).__init__()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Action meaning:
            #   0: Do nothing with the PHS
            #  -1: Use the PHS as much as we can to provide the demand of electricity
            #   1: Store as much energy as possible in the PHS
        # self.action_space = gym.spaces.Box(low=np.array([min1,min2]), high=np.array([max1,max2]), shape(2,), dtype=np.float32) # if we need more than one action


        # Collection of the data

        self.learning=learning
        
        if annee1_path=="" :
            if learning :
                self.region,self.annee1,annee1_path=selection_annee_aleatoire("./ech_apprentissage")
                self.region,self.annee2,annee2_path=selection_annee_aleatoire("./ech_apprentissage",self.region)
            else :
                self.region,self.annee1,annee1_path=selection_annee_aleatoire("./ech_test")
                self.region,self.annee2,annee2_path=selection_annee_aleatoire("./ech_test",self.region)

        else : 
            annee2_path=annee1_path
            self.annee1=2001
            self.annee2=2001 #année arbitraire
        #chargement des données pour deux années
        
        solar_path1=os.path.join(annee1_path,"solar.csv")
        wind_path1=os.path.join(annee1_path,"wind_onshore.csv")
        solar_path2=os.path.join(annee2_path,"solar.csv")
        wind_path2=os.path.join(annee2_path,"wind_onshore.csv")
        
        solar_data1 = pd.read_csv(solar_path1)['facteur_charge'].values
        wind_data1 = pd.read_csv(wind_path1)['facteur_charge'].values
        solar_data2 = pd.read_csv(solar_path2)['facteur_charge'].values
        wind_data2 = pd.read_csv(wind_path2)['facteur_charge'].values

        self.solar_data = np.concatenate([solar_data1, solar_data2])
        self.wind_data = np.concatenate([wind_data1, wind_data2])
        
        
        demand_data = pd.read_csv('./data/demand2050_ADEME.csv', header=None)
        demand_data.columns = ["time","demand"]
        self.times = demand_data['time'].values
        self.time = 0   # Time indicator
        self.ind=0 #compte les heures
        self.begin = 0
        self.end= self.times[-1]
         
        self.demand = demand_data['demand'].values
        
        #self.solar_data = pd.read_csv('./data/solar.csv')['facteur_charge'].values
        #self.wind_data = pd.read_csv('./data/wind_onshore.csv')['facteur_charge'].values

        #traitement des années bissextiles
        self.nb_jours_annee=365#+(int(self.annee1)%4==0)
        
        #repère sur l'année en cours
        self.annee1=True

        # definition of the variables of the environment
       
        #self.wind_capacity = 170.1
        #self.solar_capacity = 308.4 # max of energy we can gather using wind / sun in one step
        
        # Calcul consommation annuelle
        annual_demand = self.demand.sum()

        # Calcul production potentielle pour 1GW installé
        annual_solar_per_gw = self.solar_data.sum()
        annual_wind_per_gw = self.wind_data.sum()

        # Pour avoir 100% de surproduction et un mix 50-50
        target_production = annual_demand * 3.5
        self.wind_capacity = target_production/(2 * annual_wind_per_gw)
        self.solar_capacity = target_production/(2 * annual_solar_per_gw)


        self.phs_capacity = 180
        self.phs_power = 9.3*2
        self.phs_efficiency = 0.75
        self.gas_capacity = 125000/0.4
        
        self.gas_power_in = 7.66*2
        self.gas_power_out = 32.93*2
        self.gas_efficiency = 0.4

        # Box observation space with 5 dimensions
        #                                                     time  date   residual prod    phs level   gaz level
        self.observation_space = gym.spaces.Box(low=np.array( [0,   0,    -1,               0,          0]), 
                                                high=np.array([1,   1,     1,               1,          1]),
                                                dtype=np.float64)
        # Self observation space : State vector (date time PHS levels and so on)
        # Self state : Usually at None, used to initialize values at the very first round
        
        self.state, self.eval_data=self.reset()
        columns = ['time','action','phs_storage', 'gas_storage', 'phs_in', 'phs_used', 'gas_in', 'gas_used', 'total_energy_stored', 'furnished_demand', 'no_furnished_demand', 'residual_production', 'wasted_energy', 'stored_energy', 'reward']
        
        taille_df = self.eval_data["nb_heures"] # A VERIFIER
        self.eval_df = pd.DataFrame(None, index=range(taille_df), columns=columns)
        self.eval_df.loc[0, "phs_storage"] = self.eval_data['phs_storage']
        self.eval_df.loc[0, "gas_storage"] = self.eval_data['gas_storage']
        
        self.eval_df.loc[0, "phs_in"] = 0
        self.eval_df.loc[0, "phs_used"] = 0
        self.eval_df.loc[0, "gas_in"] = 0
        self.eval_df.loc[0, "gas_used"] = 0
        
        self.eval_df.loc[0, "furnished_demand"] = self.eval_data['total_furnished_demand']
        self.eval_df.loc[0, "no_furnished_demand"] = self.eval_data['total_no_furnished_demand']
        self.eval_df.loc[0, "time"] = self.begin
        self.eval_df.loc[0, "reward"] = self.eval_data['total_reward']
        self.eval_df.loc[0,"residual_production"]=self.state[2]*(self.wind_capacity+self.solar_capacity)
        
        self.eval_df.loc[0, "action"] = 0


    def reset(self,seed=None,options=None):
        """Reset the environment to its initial state"""
        if self.learning : # We start at a random day and hour of the year, with the residual production of the year, and the energy tanks half full
            self.begin=np.random.choice(self.times)
            self.end=np.random.choice(self.times)
            level_gas_init=np.random.uniform(0.3,0.7)
        else :
            self.begin=self.times[0]
            self.end=self.times[-1]
            level_gas_init=0.5
            
        self.time=self.begin
        
        self.total_reward = 0
        self.ind=0 #compte les heures
        residual_production=self.wind_capacity*self.wind_data[self.time] + self.solar_capacity*self.solar_data[self.time] - self.demand[self.time]
        
        self.state = np.array([self.time%24/24,int(self.time/24)%self.nb_jours_annee/self.nb_jours_annee,residual_production/(self.wind_capacity+self.solar_capacity),1/2,level_gas_init])

        # TO DO : the histogram of the values to see if normalization is pertinent
        obs=self.state

        info={"phs_storage": self.state[3],
              "gas_storage":self.state[4],
              "total_furnished_demand" :0,
              "total_no_furnished_demand":0,
              "total_reward":0,
              "nb_heures":(self.end-self.time)%(self.nb_jours_annee*24)+365+(int(self.annee2)%4==0)}
        # la demande fournie concerne uniquement la demande fournie avec les réserves
        return obs,info
    

    def reward_demand_step(self):
        reward=-self.eval_df.loc[self.ind, "no_furnished_demand"]
        return reward
    
    def reward_demand_step2(self):
        no_furnished_demand=self.eval_df.loc[self.ind, "no_furnished_demand"]
        furnished_demand=self.eval_df.loc[self.ind, "furnished_demand"]
        max_energy_available=self.phs_power+self.gas_power_out
        reward=0
        res_prod=self.state[2]*(self.wind_capacity+self.solar_capacity)
        if -res_prod>max_energy_available:
            reward=-(max_energy_available-furnished_demand)/max_energy_available
        elif -res_prod>furnished_demand:
            reward=(-res_prod-furnished_demand)/res_prod
        return reward
    
    def reward_demand_step3(self):
        reward=0 
        if self.state[2]<0 :
            no_furnished_demand=self.eval_df.loc[self.ind, "no_furnished_demand"]
            furnished_demand=self.eval_df.loc[self.ind, "furnished_demand"]
            reward=-no_furnished_demand+furnished_demand
        return reward
    
    def reward_demand_periodic(self,period):
        reward=0
        if self.ind>=period and self.ind%period==0:
            no_furnished_demand_week=self.eval_df.loc[self.ind-period:self.ind, "no_furnished_demand"].sum()
            reward=-no_furnished_demand_week
        return reward
    
    def reward_demand_periodic2(self,period): #ne fonctionne pas
        reward=0
        max_energy_available=self.phs_power+self.gas_power_out
        if self.ind>=period and self.ind%period==0:
            quantity_to_furnish=np.minimum(np.ones(period+1)*max_energy_available, np.asarray(self.eval_df.loc[self.ind-period:self.ind, "no_furnished_demand"],dtype=float))
            print(quantity_to_furnish)
            print(self.eval_df.loc[self.ind-period:self.ind, "furnished_demand"])
            reward= np.sum(quantity_to_furnish-np.asarray(self.eval_df.loc[self.ind-period:self.ind, "furnished_demand"],dtype=float))/7*24
            print(reward)
            sum_quantity_to_furnish=np.sum(quantity_to_furnish)
            if sum_quantity_to_furnish>0:
                reward=reward/sum_quantity_to_furnish
                print(reward)
        return reward
    
    def reward_demand_periodic3(self,period):
        reward=0
        if self.ind>=period and self.ind%period==0:
            no_furnished_demand=self.eval_df.loc[self.ind-period:self.ind, "no_furnished_demand"].sum()
            furnished_demand=self.eval_df.loc[self.ind-period:self.ind, "furnished_demand"].sum()
            reward=furnished_demand-no_furnished_demand
        return reward
    
    def reward_demand_end(self):
        reward=0
        if self.time==self.end :
            reward=-self.eval_df["no_furnished_demand"].sum()
        return reward
    
    def reward_demand_end2(self):
        reward=0
        if self.time==self.end :
            res_prod=np.copy(np.asarray(self.eval_df["residual_production"]))
            res_prod[res_prod>0]=0
            if np.sum(res_prod)<0:
                reward=self.eval_df["no_furnished_demand"].sum()/np.sum(res_prod)
        return reward
    
    def reward_low_level(self):
        reward=0
        if self.state[3]<0.05:
            reward=-0.05
        if self.state[4]<0.2:
            reward=-0.1
        return reward
    
    def reward_gas_storage(self):
        reward=0
        if self.eval_df.loc[self.ind, "no_furnished_demand"]>0 :
            reward=self.state[4]/10
        return reward
    
    def reward_gas_storage_end(self):
        reward=0
        if self.time==self.end :
            reward=self.state[4]*5
        return reward
    
    def bonus_rempli(self):
        reward=0
        residual_prod=(self.wind_capacity+self.solar_capacity)*self.state[2]
        if residual_prod>self.phs_power+self.gas_power_in :
            reward=((self.eval_df.loc[self.ind, "phs_in"]/self.phs_power)>0.9)*0.03
        return reward
    #reward qui prend en compte le step, la semaine et la fin
    def reward_v1(self):
        return (0.001*self.reward_demand_step()+10*self.reward_demand_end()+self.reward_demand_periodic(20*24))/(self.phs_capacity+self.gas_capacity)


    def reward_v2(self):
        return (self.reward_demand_periodic(7*24)+self.reward_demand_step())/(self.gas_capacity+self.phs_capacity)
    
    def reward_v3(self):
        return (10e-5*self.reward_demand_step()+10*self.reward_demand_end()+self.reward_demand_periodic(20*24))/10**5
    
    def reward_v4(self):
        return (0.001*self.reward_demand_step()+10*self.reward_demand_end()+self.reward_demand_periodic(24))/10**5
    
    def reward_v5(self):                           # grosse pénalité si le gaz devient très bas
        return self.reward_v1()+self.reward_gas_storage()+self.reward_gas_storage_end()
    
    def reward_v6(self):                           # grosse pénalité si le gaz devient très bas
        return self.reward_v1()+self.reward_gas_storage_end()
    
    def reward_v7(self):                           # grosse pénalité si le gaz devient très bas
        return self.reward_v1()+self.reward_gas_storage_end()*100+self.reward_gas_storage()*1000
    
    def reward_v8(self):
        return (0.001*self.reward_demand_step()+10*self.reward_demand_end()+self.reward_demand_periodic(24)*0.1+self.reward_demand_periodic(20*24))/10**5
    
    def reward_v9(self):
        return (0.01*self.reward_demand_step3()+self.reward_demand_periodic3(24))/10**5
    

    def reward_v10(self):
        return self.reward_v1()+self.reward_low_level()
    
    def reward_v11(self):
        return (0.001*self.reward_demand_step()+100*self.reward_demand_end()+self.reward_demand_periodic(20*24))/10**5
    
    def reward_v12(self):
        return self.reward_demand_step2()*2+self.reward_low_level()+self.reward_demand_end2()*10+self.bonus_rempli()
        




###############################################################   U P D A T E   L E V E L   1   #####################################################################################################

    def update_levels_guided(self,action,residual_production):
        phs_in=0 
        gas_in=0 
        phs_usage=0 
        gas_usage=0
        no_furnished_demand=0

        if action[0]>=0 :
            qty_asked_for_phs=min(self.phs_power*action[0],(1 - self.state[3])*self.phs_capacity)      # L'action est un pourcentage de la puissance max des PHS
                                                            # Si la production résiduelle est positive, on remplit les PHS autant qu'on peut avec, le reste est fourni par le Gaz
                                                            # Sinon, on fournit autant qu'on peut avec le Gaz pour satisfaire la demande.
                                                            # Si le Gaz suffit, on utilise aussi du Gaz pour remplir les PHS
                                                            # Sinon on vide les PHS pour fournir la demande
           
            if (residual_production > qty_asked_for_phs):
                phs_in = qty_asked_for_phs
                gas_in = (residual_production - phs_in) #On ajoutera les contraintes à la fin
                
            else:
                gas_usage = min(qty_asked_for_phs - residual_production, self.gas_power_out, self.state[4]*self.gas_capacity*self.gas_efficiency) #qté de gaz à sortir pour remplir les phs en complément de la prod résiduelle
                    # La quantité de Gaz que l'on va vider pour remplir les PHS et la demande
                
                if (residual_production + gas_usage > 0):
                    phs_in=residual_production+gas_usage #on ajoutera les contraintes à la fin
                    
                else :#ici on le force à vider pour répondre à la demande, j'ai peur que l'agent comprenne pas ce qui se passe => ne rien faire plutot et laisser une demande pas remplie ????
                    phs_usage=min(-(residual_production+gas_usage), self.phs_power, self.state[3]*self.phs_efficiency*self.phs_capacity)
             
                    
        else: # on vide les phs
            # rajouter un cas ou on rempli quand meme les phs
            phs_usage=min(-self.phs_power*action[0], self.state[3]*self.phs_efficiency*self.phs_capacity)

            if residual_production>min(self.gas_power_in,self.gas_capacity-self.state[4]*self.gas_capacity) : #on rempli le gaz puis les phs
                
                gas_in=min(self.gas_power_in,self.gas_capacity-self.state[4]*self.gas_capacity)
                phs_usage=0 
                phs_in=min(residual_production-gas_in,self.phs_power,self.phs_capacity-self.state[3]*self.phs_capacity)
                
            elif residual_production+phs_usage>min(self.gas_power_in,self.gas_capacity-self.state[4]*self.gas_capacity): #on ne vide pas les phs autant que demandé
                gas_in=min(self.gas_power_in,self.gas_capacity-self.state[4]*self.gas_capacity)
                phs_usage=gas_in-residual_production
                
            elif residual_production+phs_usage>0 : # On utilise tous les phs demandé mais on peut remplir un peu le gaz
                gas_in= residual_production+phs_usage
            else : #on vide aussi du gaz pour essayer de répondre à la demande
                gas_usage=min(self.state[4]*self.gas_capacity*self.gas_efficiency,self.gas_power_out,-(residual_production+phs_usage))
            
       
            
        # Adding the Constraints
        phs_in=min(phs_in,self.phs_power,self.phs_capacity-self.state[3]*self.phs_capacity)
        gas_in=min(gas_in,self.gas_power_in,self.gas_capacity-self.state[4]*self.gas_capacity)
        
        
        
        # Update of the capacity levels
        state3=self.state[3]+phs_in/self.phs_capacity-phs_usage/self.phs_efficiency/self.phs_capacity
        state4=self.state[4]+gas_in/self.gas_capacity-gas_usage/self.gas_efficiency/self.gas_capacity
        
        if residual_production<0 :
            no_furnished_demand=max(0,-residual_production-gas_usage-phs_usage+phs_in+gas_in)
        
        # For unit tests
        return (state3,state4,no_furnished_demand, phs_in, gas_in, phs_usage, gas_usage)
    

###############################################################   U P D A T E   L E V E L   2   #####################################################################################################
    
    # We do as the agent says, only physical limits can override its action
    def update_levels_unguided(self,action,residual_production):
        phs_in=0 
        gas_in=0 
        phs_usage=0 
        gas_usage=0
        no_furnished_demand=0

        if action[0]>=0 :
            qty_asked_for_phs=self.phs_power*action[0] 
             
            if (residual_production > qty_asked_for_phs):
                phs_in = min((1 - self.state[3])*self.phs_capacity, qty_asked_for_phs)
                gas_in = (residual_production - qty_asked_for_phs) #perte d'énergie si l'agent veut trop remplir les phs
                     
            else:
                gas_usage = min(qty_asked_for_phs - residual_production, self.gas_power_out, self.state[4]*self.gas_efficiency*self.gas_capacity)#, (1 - self.state[3])*self.phs_capacity-residual_production) #qté de gaz à sortir pour remplir les phs en complément de la prod résiduelle
                # La quantité de Gaz que l'on va vider pour remplir les PHS et la demande
                phs_in=min(qty_asked_for_phs, gas_usage, (1 - self.state[3])*self.phs_capacity) 
                no_furnished_demand=min(0,gas_usage-phs_in+residual_production)
         
                
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
        if residual_production<0 :
            no_furnished_demand=max(0,-residual_production-gas_usage-phs_usage+phs_in+gas_in)
        
                                                       #pour des tests unitaires
        return (state3,state4,no_furnished_demand, phs_in, gas_in, phs_usage, gas_usage)


    reward_functions = {
        "reward_v1": reward_v1,
        "reward_v2": reward_v2,
        "reward_v3": reward_v3,
        "reward_v4": reward_v4,
        "reward_v5": reward_v5,
        "reward_v6": reward_v6,
        "reward_v7": reward_v7,
        "reward_v8": reward_v8,
        "reward_v9": reward_v9,
        "reward_v10": reward_v10,
        "reward_v11": reward_v11,
        "reward_v12": reward_v12
    }
    
    update_levels_functions = {
    "update_levels_guided": update_levels_guided,
    "update_levels_unguided": update_levels_unguided
    }

    reward_function = reward_functions[reward_function]
    update_level_function = update_levels_functions[update_levels_function]


    def step(self, action):
        """Take an action and return the new state, reward, done, and info"""
        assert self.action_space.contains(action), "Invalid action!" +str(action)


        # Add later year periodicity
        # Time update at each step, The first value corresponds to the number of days
        # (a float, the decimal part representing the progress of the current day).
        # The second one is a 24hours periodic function, so the agent can better understand how days work,
        # understanding for example that every night the consumption decreases
      
        self.time = (self.time + 1) % len(self.times)
        self.state[0]=self.time%24/24 # heure
        self.state[1]=(int(self.time/24)%self.nb_jours_annee)/self.nb_jours_annee #jour
        
        
        residual_production = self.wind_capacity*self.wind_data[self.time+int(not(self.annee1))*(self.times).shape[0]] + self.solar_capacity*self.solar_data[self.time+int(not(self.annee1))*(self.times).shape[0]] - self.demand[self.time]
      
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
        #assert (action >= 0 or (action[0] < 0 and phs_in <= 0)), "L'agent demande de vider les phs et ils se remplissent"
        #test pour vérifier qu'on ne créé pas d'énergie
        assert ((residual_production <= 0) or (residual_production > 0 and round(residual_production, 6) >= round(phs_in+gas_in-phs_usage-gas_usage, 6))), "On a créé de l'énergie, vite déposez un brevet !!!"
        
        #vérification des débits
        assert (phs_in<=self.phs_power), "Le remplissage des phs est suppérieur au débit possible"
        assert (phs_usage<=self.phs_power), "L'utilisation des phs est suppérieure au débit possible"
        assert (gas_in<=self.gas_power_in), "Le remplissage du gaz est suppérieur au débit possible"
        assert (gas_usage<=self.gas_power_out), "L'utilisation des phs est suppérieure au débit possible"
        assert (round(gas_in,10)>=0),"problème de signe sur le gaz in : " +str(gas_in) + " action : " + str(action) + " production résiduelle : " + str(residual_production)
        assert (round(gas_usage,10)>=0),"problème de signe sur le gaz out : "+ str(gas_usage)+ " action : " + str(action) + " production résiduelle : " + str(residual_production)
        assert (round(phs_in,10)>=0),"problème de signe sur les phs in : " +str(phs_in)+ " action : " + str(action) + " production résiduelle : " + str(residual_production)
        assert (round(phs_usage,10)>=0),"problème de signe sur les phs out" + str(phs_usage) + " action : " + str(action) + " production résiduelle : " + str(residual_production)
        
        assert (round(self.eval_df.loc[self.ind, "no_furnished_demand"],10)>=0),"problème de signe sur la demande non fournie" + str(self.eval_df.loc[self.ind, "no_furnished_demand"])
        assert (round(self.eval_df.loc[self.ind, "furnished_demand"],10)>=0),"problème de signe sur la demande fournie" + str(self.eval_df.loc[self.ind, "no_furnished_demand"])
      
        
        # Reward function
        # cout basé sur la pénalité d'écrétage / de perte par conversion d'énergie ?
        
        

        # info : evaluation data
        
        self.eval_data["phs_storage"]=self.state[3]
        self.eval_data["gas_storage"]=self.state[4]
        
        if residual_production<0 :
            self.eval_data["total_furnished_demand"]-=(residual_production+no_furnished_demand)
        self.eval_data["total_no_furnished_demand"]+=no_furnished_demand
        
        
        self.ind+=1
        self.eval_df.loc[self.ind, "phs_storage"] = self.state[3]
        self.eval_df.loc[self.ind, "gas_storage"] = self.state[4]
        self.eval_df.loc[self.ind, "furnished_demand"] = max(0,-residual_production-no_furnished_demand)
        self.eval_df.loc[self.ind, "no_furnished_demand"] = no_furnished_demand
        self.eval_df.loc[self.ind, "time"] = self.time
        self.eval_df.loc[self.ind,"residual_production"]=residual_production
        
        self.eval_df.loc[self.ind, "phs_in"] = phs_in
        self.eval_df.loc[self.ind, "phs_used"] = phs_usage
        self.eval_df.loc[self.ind, "gas_in"] = gas_in
        self.eval_df.loc[self.ind, "gas_used"] = gas_usage
        self.eval_df.loc[self.ind, "action"] = action
        
        reward = self.reward_function()
        
        self.eval_data["total_reward"]+=reward
        
        self.eval_df.loc[self.ind, "reward"] = reward

        

        
        

        

        # Termination condition
        terminated = bool(self.time%(365*24)==self.end) and not(self.annee1)
        
        if bool(self.time==(self.begin-1)%(365*24)) and self.annee1 :
            self.annee1=False
        
        
        
        truncated=False

        return self.state, reward, terminated, truncated, self.eval_data


    def render(self, mode='human'):
        """Render the environment (optional)"""
        if not(self.learning) and self.time==self.end :
            print(self.eval_data)
            
            
            #plot de la fourniture de la demande
            plt.figure(figsize=(20,10))
            plt.subplot(311)
            plt.plot(self.eval_df["phs_storage"], label='Niveau PHS')
            plt.legend()
            plt.title('Niveau de stockage phs')
            
            plt.subplot(312)
            plt.plot(self.eval_df["gas_storage"], label='Niveau P2G')
            plt.legend()
            plt.title('Niveaux de stockage gaz')
            
            plt.subplot(313)
            plt.plot(self.eval_df["no_furnished_demand"], label='Dem no fournie')
            plt.legend()
            plt.title('Demande non fournie')
            plt.show()
            
            plt.figure(figsize=(20,10))
            plt.plot(self.eval_df["residual_production"], label='res_prod')
            plt.plot(-self.eval_df["no_furnished_demand"], label='Déficit')
            plt.legend()
            plt.title('Production résiduelle et déficits horaires')
            plt.show()
            
            # Prepare data for the chart
            
            ind=24
            T=self.eval_df.loc[ind, "time"]
            
            #stack_data = [self.wind_data[T:T+168]*self.wind_capacity, self.solar_data[T:T+168]*self.solar_capacity, self.eval_df.loc[ind:ind+167, "phs_used"].values , self.eval_df.loc[ind:ind+167, "gas_used"].values]
            stack_data = [np.asarray(self.wind_data[T:T+168]*self.wind_capacity, dtype=float), np.asarray(self.solar_data[T:T+168]*self.solar_capacity, dtype=float), np.asarray(self.eval_df["phs_used"].iloc[ind:ind+168].values, dtype=float), np.asarray(self.eval_df["gas_used"].iloc[ind:ind+168].values, dtype=float),np.asarray(self.eval_df["no_furnished_demand"].iloc[ind:ind+168].values, dtype=float)]
            labels = ['Offshore + Onshore', 'PV', 'PHS', 'Methanation', "no_furnished_demand"]

            # Create the chart
            plt.figure(figsize=(16, 8))
            plt.stackplot(np.arange(T,T+168), stack_data, labels=labels, alpha=0.8)

            # Add demand
            plt.plot(np.arange(T,T+168), self.demand[T:T+168], color='black', linewidth=2, label='Demand')

            # Add legend and labels
            plt.title('Cumulative Energy Production', fontsize=16)
            plt.ylabel('Production (GWh)', fontsize=12)
            plt.xlabel('Time (hours)', fontsize=12)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Display the chart
            plt.show()
            
            
            ind=1000
            T=self.eval_df.loc[ind, "time"]
            
            #stack_data = [self.wind_data[T:T+168]*self.wind_capacity, self.solar_data[T:T+168]*self.solar_capacity, self.eval_df.loc[ind:ind+167, "phs_used"].values , self.eval_df.loc[ind:ind+167, "gas_used"].values]
            stack_data = [np.asarray(-self.eval_df["phs_used"].iloc[ind:ind+168].values, dtype=float), np.asarray(-self.eval_df["gas_used"].iloc[ind:ind+168].values, dtype=float),np.asarray(self.eval_df["phs_in"].iloc[ind:ind+168].values, dtype=float), np.asarray(self.eval_df["gas_in"].iloc[ind:ind+168].values, dtype=float), -np.asarray(self.eval_df["no_furnished_demand"].iloc[ind:ind+168].values, dtype=float)]
            labels = ['phs_used', 'gas_used','phs_in', 'gas_in', "no_furnished_demand"]

            # Create the chart
            plt.figure(figsize=(16, 8))
            plt.stackplot(np.arange(T,T+168), stack_data, labels=labels, alpha=0.8)

            # Add demand
            plt.plot(np.arange(T,T+168), np.asarray(self.eval_df["residual_production"].iloc[ind:ind+168].values, dtype=float), color='black', linewidth=2, label='residual production')
            
            #add action
            plt.plot(np.arange(T,T+168), np.asarray(self.eval_df["action"].iloc[ind:ind+168].values, dtype=float)*10, color='grey', linewidth=2, label='action*10')

            # Add legend and labels
            plt.title('Cumulative Energy Production', fontsize=16)
            plt.ylabel('Production (GWh)', fontsize=12)
            plt.xlabel('Time (hours)', fontsize=12)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Display the chart
            plt.show()

    def close(self):
        """Clean up resources (optional)"""

check_env(CustomEnv(), warn=True, skip_render_check=True)   

#1 an cyclique, début n'importe quand dans l'année, on ne dure pas toujours 1 an, qqté non fournie

#critère de réussite : combien il a fourni, combien il a stocké
#évaluation sur une quarantaine de séries
#regarder si stable baseline peut évaluer sur un échantillon d'entrainement
#retourner un dict avec les indicateurs qui nous intéressent
#création config

#voir RTE

#graph reward
#evaluer tous les n épisodes vs heuristique à la fois pour général et overfitting