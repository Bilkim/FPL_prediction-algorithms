# Imports
import requests, json
import pandas as pd
import numpy as np
# from pprint import pprint
# import matplotlib.pyplot as plt
import seaborn as sns
import re
from pprint import pprint
from pulp import *


sns.set_style('whitegrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)   

# base url for all FPL API endpoints
base_url = 'https://fantasy.premierleague.com/api/'

# get data from bootstrap-static endpoint
r = requests.get(base_url+'bootstrap-static/').json()

# show the top level fields
# pprint(r, indent=2, depth=1, compact=True)

r = requests.get(base_url+'bootstrap-static/').json()

df = pd.json_normalize(r['elements'])

# Loading the json data and making the dataframe
data = r

player_data_json = data['elements']
pdata = pd.json_normalize(player_data_json)
to_drop = ['chance_of_playing_this_round','chance_of_playing_next_round','code','cost_change_event','cost_change_event_fall','cost_change_start','cost_change_start_fall','dreamteam_count','ep_this','event_points','form','ict_index','in_dreamteam','news','photo','special','squad_number','status','transfers_in','transfers_in_event','transfers_out','transfers_out_event','value_form','value_season']
pdata.drop(to_drop, axis=1, inplace = True)
pdata['full_name'] = pdata.first_name + " " + pdata.second_name
pdata['element_type_name'] = pdata.element_type.map({x['id']:x['singular_name_short'] for x in data['element_types']})
pdata = pdata.loc[:,['full_name','first_name','second_name', 'element_type','element_type_name','id','team', 'team_code', 'web_name',
                     'saves','penalties_saved','clean_sheets','goals_conceded',
                     'bonus', 'bps','creativity','ep_next','influence', 'threat',
                     'goals_scored','assists','minutes', 'own_goals',
                     'yellow_cards', 'red_cards','penalties_missed',
                     'selected_by_percent', 'now_cost','points_per_game','total_points']]
pdata['team'] = pdata.team.map({x['id']:x['name'] for x in data['teams']})




impute_cols = ['saves','penalties_saved', 'clean_sheets', 'goals_conceded', 'bonus', 'bps',
               'creativity', 'influence', 'threat', 'goals_scored','assists', 'minutes', 'own_goals',
               'yellow_cards', 'red_cards','penalties_missed','points_per_game', 'total_points']
positions = set(pdata.element_type_name)
costs = set(pdata.now_cost)
medians = {}; stds = {}

for i in positions:
    medians['{}'.format(i)] = {}
    stds['{}'.format(i)] = {}
    for c in costs:
        medians['{}'.format(i)]['{}'.format(c)] = {}
        stds['{}'.format(i)]['{}'.format(c)] = {}
        for j in impute_cols:
            if pdata[(pdata.total_points!=0)&(pdata.minutes!=0)&(pdata.element_type_name==str(i))&(pdata.now_cost==c)].shape[0] > 0:
                median = np.median(pdata[(pdata.total_points!=0)&(pdata.minutes!=0)&(pdata.element_type_name==i)&(pdata.now_cost==c)][j].astype(np.float32))
                std = np.std(pdata[(pdata.total_points!=0)&(pdata.minutes!=0)&(pdata.element_type_name==i)&(pdata.now_cost==c)][j].astype(np.float32))
                medians['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = median
                stds['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = std
            else:
                medians['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = 0
                stds['{}'.format(i)]['{}'.format(c)]['{}'.format(j)] = 0

for idx, row in pdata[(pdata.total_points==0)&(pdata.minutes==0)].iterrows():
    for col in impute_cols:
        pdata.loc[idx,col] = medians[str(row['element_type_name'])][str(row['now_cost'])][str(col)] + np.abs((np.random.randn()/1.5)*stds[str(row['element_type_name'])][str(row['now_cost'])][str(col)])
        
        
print(pdata.columns)
prob = pulp.LpProblem('FantasyTeam', LpMaximize)

# Create the decision variables.. all the players
def create_dec_var(pdata):
    decision_variables = []
    for rownum, row in pdata.iterrows():
        variable = str('x' + str(rownum))
        variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat= 'Integer') #make variables binary
        decision_variables.append(variable)

    return decision_variables

# This is what we want to maximize (objective function)
def total_points(pdata,decision_variables,prob):
    total_points = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                formula = row['total_points']*player
                total_points += formula

    prob += total_points
    # print ("Optimization function: " + str(total_points))
    
    return prob

# Add constraint for cash
def cash(pdata,decision_variables,prob,avail_cash):
    avail_cash = avail_cash
    total_paid = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                formula = row['now_cost']*player
                total_paid += formula

    prob += (total_paid <= avail_cash)
    return prob

# Add constraint for number of goalkeepers
def team_gkp(pdata,decision_variables,prob,avail_gk):
    avail_gk = avail_gk
    total_gk = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:                
                if row['element_type'] == 1:
                    formula = 1*player
                    total_gk += formula
    prob += (total_gk == avail_gk)
    return prob



# Add constraint for number of defenders
def team_def(pdata,decision_variables,prob,avail_def):
    avail_def = avail_def
    total_def = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['element_type'] == 2:
                    formula = 1*player
                    total_def += formula
    prob += (total_def == avail_def)
    return prob


# Add constraint for number of midfielders
def team_mid(pdata,decision_variables,prob,avail_mid):
    avail_mid = avail_mid
    total_mid = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['element_type'] == 3:
                    formula = 1*player
                    total_mid += formula
    prob += (total_mid == avail_mid)
    return prob

def team_fwd(pdata,decision_variables,prob,avail_fwd):    # Forward Constraint
    avail_fwd = avail_fwd
    total_fwd = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['element_type'] == 4:
                    formula = 1*player
                    total_fwd += formula
    prob += (total_fwd == avail_fwd)
    return prob

def team_max_players(pdata, decision_variables,prob):
    team_dict= {}
    for team in set(pdata.team_code):
        team_dict[str(team)]=dict()
        team_dict[str(team)]['avail'] = 3
        team_dict[str(team)]['total'] = 0
        for rownum, row in pdata.iterrows():
            for i, player in enumerate(decision_variables):
                if rownum == i:
                    if row['team_code'] == team:
                        formula = 1*player
                        team_dict[str(team)]['total'] += formula

        prob += (team_dict[str(team)]['total'] <= team_dict[str(team)]['avail'])
        
        return prob


def LP_optimize(pdata, prob):   
    prob.writeLP('FantasyTeam.lp')
    optimization_result = prob.solve()
    assert optimization_result == LpStatusOptimal
    # print("Status:", LpStatus[prob.status])
    # print("Optimal Solution to the problem: ", value(prob.objective))
    # print ("Individual decision_variables: ")
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)
        
        
# Assemble the whole problem data
def find_prob(df,ca,gk,de,mi,fw,pr):
    lst = create_dec_var(df)
    
    prob = total_points(df,lst,pr)
    prob = cash(df,lst,prob,ca)
    prob = team_gkp(df,lst,prob,gk)
    prob = team_def(df,lst,prob,de)
    prob = team_mid(df,lst,prob,mi)
    prob = team_fwd(df,lst,prob,fw)
    prob = team_max_players(df,lst,prob)
    
    return prob
 

def df_decision(pdata, prob):  
    variable_name = []
    variable_value = []

    for v in prob.variables():
        variable_name.append(v.name)
        variable_value.append(v.varValue)

    df = pd.DataFrame({'variable': variable_name, 'value': variable_value})
    for rownum, row in df.iterrows():
        value = re.findall(r'(\d+)', row['variable'])
        df.loc[rownum, 'variable'] = int(value[0])

    df = df.sort_values(by='variable')

    #append results
    for rownum, row in pdata.iterrows():
        for results_rownum, results_row in df.iterrows():
            if rownum == results_row['variable']:
                pdata.loc[rownum, 'decision'] = results_row['value']

    pdata[pdata.decision==1].now_cost.sum() # Returns 830
    total_estimate_points = pdata[pdata.decision==1].total_points.sum() # Returns 2010.8606251232461
    data = pdata[pdata.decision==1].sort_values('element_type').head(11)
    
    data.to_csv("541.csv")
    
    
    
    return total_estimate_points

# prob_m = find_prob(df,830,1,3,4,3,prob)
# prob_m = find_prob(df,830,1,3,5,2,prob)
# prob_m = find_prob(df,830,1,4,3,3,prob)
# prob_m = find_prob(df,830,1,4,4,2,prob)
# prob_m = find_prob(df,830,1,4,5,1,prob)
# prob_m = find_prob(df,830,1,5,3,2,prob)
prob_m = find_prob(df,830,1,5,4,1,prob)

LP_optimize(pdata, prob_m)

result = df_decision(pdata, prob_m)

with open('fpl_points.txt', 'w') as f:
    f.write(str(result))
    f.write('\n')


