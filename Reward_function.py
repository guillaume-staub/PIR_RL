# Reward function for the GAS / PHS case

# Our main goal is: to satisfy the demand every day while ending with tanks at least as full as they were at the start

# To do so, we have:

    # To satisfy the demand "at all costs", and if we cannot, we have to make sure the agent tries its best to limitate the unfurnished amount of electricity

        # ==> bonus if the demand is satisfied at each hour
# Bonus: fixed, at each step
        # ==> malus depending on the unsatisfied demand each hour, needs to be adjusted with coefficients to be sure it understands the main goal is to satisfy the demand and not to store energy
# Malus: depending on the unsatisfied demands, at each step




    #   Not to waste energy / store energy every time we can
        
        #   ==> Bonus depending on how much it stored each hour, if it used all the renewable energy at disposition before using GAS / PHS
# Bonus: dependant on the energy stored, at each step
        #   ==> Malus if we notice that it lost energy by doing wrong actions (like emptying the PHS when we had enough renewable energy for the hour)
# Malus: dependant on the wasted energy, at each step




    #   Keep the level of tanks as high as possible:
    #       it does not know when the period of evaluation ends, so we need to periodically check the levels to be sure he does not use them fully without "thinking" at what comes next

        #   ==> Bonus each week if the level of the tanks rose compared to the previous week
        #   ==> Malus if they decreased. However the malus should be nullified, or at least less high than the bonus it receives if it spent the week satisfying the demand and not wasting energy,
        #       as this probably means it had no other choice
# Bonus / Malus: dependant on the difference of tanks fulness' levels at a week difference, each week (168 steps)
        #   ==> Bonus / Malus at the end of the period if the level of the tanks are higher than what they were at the start:
        #       The period should be long enough to allow it to store more energy than it has to spend if it acts wisely
# Bonus / Malus: dependant on the difference of tanks fulness' levels from begin to end, once a period (total steps)



# We need to register every round some data of the trajectory (let's call it trajectory_data for now)
#   The levels of the tanks at each step : 'PHS Level' and 'GAS Level'
#   The total of the two (not between 0 and 1 but the total energy stored in the tanks) 'Total Energy Stored'
#   The amount of energy stored and wasted at each step 'Stored Energy', 'Wasted Energy'
#   The amount of unfurnished energy 'No Furnished Demand'
#   The 


import numpy as np

def reward_v3(self, no_furnished_demand, wasted_energy, stored_energy, trajectory_data, base_reward = 1):
    reward = 0
    # hourly rewards:
        # Unfurnished demand:
    if (no_furnished_demand <= 0):
        reward -= no_furnished_demand           # See the scale of values taken and add coefficient to balance it (in parameters function ? to automate and try with different coefficient)
    else:
        reward += base_reward                   # The smallest amount of reward granted, used to scale the other

        # Energy wasted and stored: 
    if (wasted_energy > 0):
        reward -= wasted_energy
    else:
        reward += stored_energy

    # Weekly rewards
        # Keeping the levels of the tanks high:
    if (self.time%168 == 0 and self.time != 0): # Every week (168 hours/steps)
        if(trajectory_data['Total Energy Stored'][self.time] - trajectory_data['Total Energy Stored'][self.time - 168] > 0): # We managed to store more energy than 168 hours ago
            reward += trajectory_data['Total Energy Stored'][self.time] - trajectory_data['Total Energy Stored'][self.time - 168] # May be too big depending on the week, should probably be coefficiented (sigmoid ?)
        else: # We lost energy, now we need to see what are the circumstances to decide whether to punish the agent or not
            bad_actions = np.sum(trajectory_data['Wasted Energy'][self.time - 168:]) + np.sum(trajectory_data['Unfurnished Demand'][self.time - 168:])
            mean_residual_production = (1/7)*np.sum(trajectory_data['Residual Production'][self.time - 168]) # Mean of residual production per day over the last week
            if(bad_actions >= mean_residual_production):
                reward -= trajectory_data['Total Energy Stored'][self.time] - trajectory_data['Total Energy Stored'][self.time - 168]
                
    return reward