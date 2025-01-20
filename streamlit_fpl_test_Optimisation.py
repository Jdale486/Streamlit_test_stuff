##import packages

import requests
import json
import pandas as pd
import numpy as np
from decimal import Decimal
import itertools as it
import streamlit as st

number = st.number_input("Enter Team ID", step=1)
st.write("The current Team ID is ", number)

##Request fixtures from FPL API

url = f"https://fantasy.premierleague.com/api/fixtures/"
response = requests.get(url)
json_data = response.json()

Fixtures_Dim = json_data

Fixtures_Dim = pd.json_normalize(Fixtures_Dim)


Fixtures_Dim_Future=Fixtures_Dim[Fixtures_Dim['finished']==False]

### Used as variable to return only the data for completed gameweeks

UpcomingWeek = Fixtures_Dim_Future['event'].min().astype(int)
PrevWeek = UpcomingWeek-1

### Variable for number of gameweeks to go back to calculate averages from

NoWeeks = 4

TeamID = number

#TeamID = 1769272

print(UpcomingWeek)
print(PrevWeek)

###Request Team Data from FPL API

url = f"https://fantasy.premierleague.com/api/bootstrap-static"
response = requests.get(url)
json_data = response.json()

teams_dim = json_data['teams']

Teams_Dim = pd.json_normalize(teams_dim)

###Duplicates of Fixtures for Home and Away, so that this can be merged with elements

Fixtures_Away1 = Fixtures_Dim_Future
Fixtures_Away1.head()

Fixtures_Home1 = Fixtures_Dim_Future
Fixtures_Home1.head()

FixturesConcat = pd.concat([Fixtures_Away1,Fixtures_Home1], keys=["Away", "Home"])

FixturesConcat.reset_index(inplace=True)

FixturesConcat['TeamID'] = FixturesConcat.apply(
    lambda row: row['team_a'] if row['level_0'] == 'Away' else row['team_h'], 
    axis=1
)

FixturesConcat['OppTeam2'] = FixturesConcat.apply(
    lambda row: row['team_a'] if row['level_0'] == 'Home' else row['team_h'], 
    axis=1
)

### Gameweek elements data from FPL API

player_data = []

for gw1 in range(1,UpcomingWeek):
    url = f"https://fantasy.premierleague.com/api/event/{gw1}/live/"
    
    response = requests.get(url)
    json_data = response.json()
    print(url)

    for gw in json_data['elements']:
        #i1 = gw['id']
        playerid=gw['id']
        minutes=gw['stats']['minutes']
        goals_scored=gw['stats']['goals_scored']
        assists=gw['stats']['assists']
        clean_sheets=gw['stats']['clean_sheets']
        goals_conceded=gw['stats']['goals_conceded']
        own_goals=gw['stats']['own_goals']
        penalties_saved=gw['stats']['penalties_saved']
        penalties_missed=gw['stats']['penalties_missed']
        yellow_cards=gw['stats']['yellow_cards']
        red_cards=gw['stats']['red_cards']
        saves=gw['stats']['saves']
        bonus=gw['stats']['bonus']
        bps=gw['stats']['bps']
        influence=gw['stats']['influence']
        creativity=gw['stats']['creativity']
        threat=gw['stats']['threat']
        ict_index=gw['stats']['ict_index']
        starts=gw['stats']['starts']
        expected_goals=gw['stats']['expected_goals']
        expected_assists=gw['stats']['expected_assists']
        expected_goal_involvements=gw['stats']['expected_goal_involvements']
        expected_goals_conceded=gw['stats']['expected_goals_conceded']
        total_points=gw['stats']['total_points']
        if gw['explain']: 
            fixtureid = gw['explain'][0]['fixture'] 
        else: 
            fixtureid = None

        player_data.append({
            'playerid': playerid,
            'minutes': minutes,
            'goals_scored': goals_scored,
            'assists': assists,
            'clean_sheets': clean_sheets,
            'goals_conceded': goals_conceded,
            'threat': threat,
            'expected_goals': expected_goals,
            'expected_assists': expected_assists,
            'expected_goal_involvements': expected_goal_involvements,
            'expected_goals_conceded': expected_goals_conceded,
            'fixtureid': fixtureid  


            })

    GWElementsDF = pd.DataFrame(player_data)
GWElementsDF.head(5)

GWElementsDF1 = pd.merge(GWElementsDF, Fixtures_Dim,left_on='fixtureid', right_on='id', how='outer')

GWElementsDF1.to_csv('FPL_Event_By_Week.csv')
GWElementsDF1.head()

###Standard Element Data from FPL API

url = f"https://fantasy.premierleague.com/api/bootstrap-static"
response = requests.get(url)
json_data = response.json()

elements_dim = json_data['elements']

Elements_Dim1 = pd.json_normalize(elements_dim)

Elements_Dim1.to_csv('elementsdim.csv')

Elements_Dim1 = pd.merge(GWElementsDF1, Elements_Dim1,left_on='playerid', right_on='id', how='outer',suffixes=('.gw', '.info'))

Elements_Dim1['ExcludeDef'] = Elements_Dim1.apply(
    lambda row: 'Y' if (row['element_type'] == '1' or row['element_type'] == '2') and row['minutes_x'] > 59 else 'N', 
    axis=1
)

Elements_Dim1.to_csv('elements4wk.csv')

###Arbitrary team weightings calculated based on average of last season's odds

TeamWeightings = {
    "Weighting": [
        1.699153895, 1.68275948, 1.639154108, 1.626396476, 1.605876872, 1.542304678,
        1.526424188, 1.50219836, 1.499125082, 1.489893392, 1.44492154, 1.424635514,
        1.365324148, 1.288771283, 1.266010528, 1.251813146, 1.18226904, 1.7, 1.7, 1.7
    ],
    "Teams.id": [
        3, 7, 16, 20, 4, 9, 8, 2, 15, 19, 5, 14, 6, 12, 18, 1, 13, 10, 11, 17
    ],
    "Teams.name": [
        "Bournemouth", "Crystal Palace", "Nott'm Forest", "Wolves", "Brentford", "Fulham",
        "Everton", "Aston Villa", "Newcastle", "West Ham", "Brighton", "Man Utd", "Chelsea",
        "Liverpool", "Spurs", "Arsenal", "Man City", "Ipswich", "Leicester", "Southampton"
    ]
}

TeamWeightings = pd.DataFrame(TeamWeightings)

Elements_Dim1['OppTeam'] = Elements_Dim1.apply(lambda row: row['team_h'] if row['team_h'] == row['team'] else row['team_a'], axis=1)

Elements_Dim2 = pd.merge(Elements_Dim1, TeamWeightings,left_on='OppTeam', right_on='Teams.id', how='outer')

Elements_Dim2['XGC_Weighted'] = Elements_Dim2['Weighting'].astype(float)*Elements_Dim2['expected_goals_conceded.gw'].astype(float)

Elements_Dim2['XGI_Weighted'] = Elements_Dim2['expected_goal_involvements.gw'].astype(float)/Elements_Dim2['Weighting'].astype(float)

###Filtered to only include the previous 4 weeks' data before it is grouped to return total sum for each team 

Elements_Dim2Filtered = Elements_Dim2[Elements_Dim2['event']>PrevWeek-4]

Elements_Dim2Grouped1 = Elements_Dim2Filtered.groupby(['OppTeam', 'event']).agg(
    XGI_Weighted=('XGI_Weighted', 'sum'),
    XGC_Weighted=('XGC_Weighted', 'max'),
    Minutes=('minutes_x', 'sum')
).reset_index()



##Average of the above taken for each team
Elements_Dim2Grouped2 = Elements_Dim2Grouped1.groupby(['OppTeam']).agg(
    XGI_Weighted=('XGI_Weighted', 'mean'),
    XGC_Weighted=('XGC_Weighted', 'mean'),
    Minutes=('Minutes', 'mean')
).reset_index()



Elements_Dim2Grouped2.head()

##Previous 4 Gameweeks this dataframe is returning the averages for the selected players 
Elements_Dim1 = Elements_Dim1[Elements_Dim1['event']<UpcomingWeek]

GWElementsDFLast4Weeks = Elements_Dim1[Elements_Dim1['event']>UpcomingWeek-NoWeeks-1]

GWElementsDFLast4Weeks = GWElementsDFLast4Weeks[['minutes_x','playerid','web_name','element_type','team','now_cost','expected_goals.gw','expected_assists.gw','expected_goal_involvements.gw','expected_goals_conceded.gw']]

GWElementsDFLast4Weeks['expected_goals.gw'] = GWElementsDFLast4Weeks['expected_goals.gw'].astype(float)
GWElementsDFLast4Weeks['expected_assists.gw'] = GWElementsDFLast4Weeks['expected_assists.gw'].astype(float)
GWElementsDFLast4Weeks['expected_goal_involvements.gw'] = GWElementsDFLast4Weeks['expected_goal_involvements.gw'].astype(float)
GWElementsDFLast4Weeks['expected_goals_conceded.gw'] = GWElementsDFLast4Weeks['expected_goals_conceded.gw'].astype(float)

GWElementsDFLast4Weeks = GWElementsDFLast4Weeks.groupby(['playerid','web_name','element_type','now_cost','team'], as_index=False)[['minutes_x','expected_goals.gw','expected_assists.gw','expected_goal_involvements.gw','expected_goals_conceded.gw']].mean(numeric_only=True)

GWElementsDFLast4Weeks['ExcludeDef'] = GWElementsDFLast4Weeks.apply(
    lambda row: 'Y' if (row['element_type'] == 1 or row['element_type'] == 2) and row['minutes_x']< 59 else 'N', 
    axis=1
)

grouped_dfLast4Weeks=GWElementsDFLast4Weeks[GWElementsDFLast4Weeks['ExcludeDef']!='Y']
grouped_dfLast4Weeks.head()

### Returns combinations of historic XG stats from players multiplied by historic opposition team performance

OpportunitiesCalc = pd.merge(grouped_dfLast4Weeks, Elements_Dim2Grouped2,left_on='team', right_on='OppTeam', how='outer')

OpportunitiesCalc['AttackOpp2'] = OpportunitiesCalc['XGC_Weighted']*OpportunitiesCalc['expected_goal_involvements.gw']

OpportunitiesCalc['DefenceOpp2'] = OpportunitiesCalc['XGI_Weighted']*OpportunitiesCalc['expected_goal_involvements.gw']

OpportunitiesCalc.sort_values(by=['AttackOpp2'])

##### Returns Output 1 FixturesElementsAll
FixturesElementsAll1 = pd.merge(FixturesConcat, Elements_Dim2Grouped2,left_on='OppTeam2', right_on='OppTeam', how='outer')

FixturesElementsAll2 = pd.merge(grouped_dfLast4Weeks, FixturesElementsAll1,left_on='team', right_on='TeamID', how='outer')

FixturesElementsAll2['AttackOpp_pergame'] = FixturesElementsAll2['XGC_Weighted']*FixturesElementsAll2['expected_goal_involvements.gw']

FixturesElementsAll2['DefenceOpp_pergame'] = FixturesElementsAll2['XGI_Weighted']*FixturesElementsAll2['expected_goals_conceded.gw']

FixturesElementsAll2Att=FixturesElementsAll2[FixturesElementsAll2['element_type']>2]

# Calculate total AttackOpp_pergame for each player
FixturesElementsAll2Att = FixturesElementsAll2Att[FixturesElementsAll2Att['event']<UpcomingWeek+4]
FixturesElementsAll2_totalsAtt = FixturesElementsAll2Att.groupby('web_name')['AttackOpp_pergame'].sum().reset_index()

# Sort players by total AttackOpp_pergame and select top 20
top_players = FixturesElementsAll2_totalsAtt.sort_values(by='AttackOpp_pergame', ascending=False).head(10)

# Filter original DataFrame to include only top 20 players
FixturesElementsAll2AttFiltered = FixturesElementsAll2Att[FixturesElementsAll2Att['web_name'].isin(top_players['web_name'])]

FixturesElementsAll2AttFilteredslim = FixturesElementsAll2AttFiltered[['web_name','event','AttackOpp_pergame']]
FixturesElementsAll2AttFilteredslim['AttackOpp_pergame'].astype(float)
FixturesElementsAll2AttFilteredslim.sort_values(by='event', ascending=True)

FixturesElementsAll2AttFilteredslim=FixturesElementsAll2AttFilteredslim.drop_duplicates(subset=['event','web_name'])

#figAttAtt = px.bar(data_frame=FixturesElementsAll2AttFilteredslim, x="event", y="AttackOpp_pergame", barmode='group', color='web_name')

st.write("Attackers Rating All")

st.bar_chart(data=FixturesElementsAll2AttFilteredslim, x="event", y="AttackOpp_pergame",color="web_name", horizontal=False, stack=False)

FixturesElementsAll2Def=FixturesElementsAll2[FixturesElementsAll2['element_type']<3]

# Calculate total DefackOpp_pergame for each player
FixturesElementsAll2Def = FixturesElementsAll2Def[FixturesElementsAll2Def['event']<UpcomingWeek+4]
FixturesElementsAll2_totalsDef = FixturesElementsAll2Def.groupby('web_name')['DefenceOpp_pergame'].sum().reset_index()

# Sort players by total DefackOpp_pergame and select top 20
top_players = FixturesElementsAll2_totalsDef.sort_values(by='DefenceOpp_pergame', ascending=True).head(10)

# Filter original DataFrame to include only top 20 players
FixturesElementsAll2DefFiltered = FixturesElementsAll2Def[FixturesElementsAll2Def['web_name'].isin(top_players['web_name'])]

FixturesElementsAll2DefFilteredslim = FixturesElementsAll2DefFiltered[['web_name','event','DefenceOpp_pergame']]
FixturesElementsAll2DefFilteredslim['DefenceOpp_pergame'].astype(float)
FixturesElementsAll2DefFilteredslim.sort_values(by='event', ascending=True)

FixturesElementsAll2DefFilteredslim=FixturesElementsAll2DefFilteredslim.drop_duplicates(subset=['event','web_name'])

#figDefDef = px.bar(data_frame=FixturesElementsAll2DefFilteredslim, x="event", y="DefenceOpp_pergame", barmode='group', color='web_name')

st.write("Defenders Defensive Rating All - Lower is Better")

st.bar_chart(data=FixturesElementsAll2DefFilteredslim, x="event", y="DefenceOpp_pergame",color="web_name", horizontal=False, stack=False)


FixturesElementsAll2Def1=FixturesElementsAll2[FixturesElementsAll2['element_type']<3]

# Calculate total Def1ackOpp_pergame for each player
FixturesElementsAll2Def1 = FixturesElementsAll2Def1[FixturesElementsAll2Def1['event']<UpcomingWeek+4]
FixturesElementsAll2_totalsDef1 = FixturesElementsAll2Def1.groupby('web_name')['AttackOpp_pergame'].sum().reset_index()

# Sort players by total Def1ackOpp_pergame and select top 20
top_players = FixturesElementsAll2_totalsDef1.sort_values(by='AttackOpp_pergame', ascending=False).head(10)

# Filter original DataFrame to include only top 20 players
FixturesElementsAll2Def1Filtered = FixturesElementsAll2Def1[FixturesElementsAll2Def1['web_name'].isin(top_players['web_name'])]

FixturesElementsAll2Def1Filteredslim = FixturesElementsAll2Def1Filtered[['web_name','event','AttackOpp_pergame']]
FixturesElementsAll2Def1Filteredslim['AttackOpp_pergame'].astype(float)
FixturesElementsAll2Def1Filteredslim.sort_values(by='event', ascending=True)

FixturesElementsAll2Def1Filteredslim=FixturesElementsAll2Def1Filteredslim.drop_duplicates(subset=['event','web_name'])

#figDefAtt = px.bar(data_frame=FixturesElementsAll2Def1Filteredslim, x="event", y="AttackOpp_pergame", barmode='group', color='web_name')

st.write("Defenders Attacking Rating All")

st.bar_chart(data=FixturesElementsAll2Def1Filteredslim, x="event", y="AttackOpp_pergame",color="web_name", horizontal=False, stack=False)


###Current Team Stuff 

CSurl = f"https://fantasy.premierleague.com/api/entry/{TeamID}/event/{PrevWeek}/picks/"
CSurl
response = requests.get(CSurl)
json_data = response.json()

TeamSelection = json_data['picks']

TeamSelection = pd.json_normalize(TeamSelection)

TeamSelection.head()

#st.plotly_chart(figDefAtt, use_container_width=True)

# Calculate total AttackOpp_pergame for each player
FixturesElementsAll3 = FixturesElementsAll2[FixturesElementsAll2['event']<UpcomingWeek+4]

FixturesElementsAll3 = pd.merge(FixturesElementsAll3, TeamSelection,left_on='playerid', right_on='element', how='outer')

FixturesElementsAll3['IsInCurrentTeam'] = np.where(
    FixturesElementsAll3['is_captain'].notnull(),
    'Y',
    'N'
)

FixturesElementsAll3FilteredAtt=FixturesElementsAll3[FixturesElementsAll3['IsInCurrentTeam']=='Y']

FixturesElementsAll3FilteredAtt=FixturesElementsAll3FilteredAtt[FixturesElementsAll3FilteredAtt['element_type_x']>2]

FixturesElementsAll3FilteredAttslim = FixturesElementsAll3FilteredAtt[['web_name','event','AttackOpp_pergame']]
#FixturesElementsAll3Filteredslim['event'].astype(int)
FixturesElementsAll3FilteredAttslim['AttackOpp_pergame'].astype(float)
FixturesElementsAll3FilteredAttslim.sort_values(by='event', ascending=True)

FixturesElementsAll3FilteredAttslim=FixturesElementsAll3FilteredAttslim.drop_duplicates(subset=['event','web_name'])


st.bar_chart(data=FixturesElementsAll3FilteredAttslim, x="event", y="AttackOpp_pergame",color="web_name", horizontal=False, stack=False)

##Current Team Defence

FixturesElementsAll3FilteredDef=FixturesElementsAll3[FixturesElementsAll3['IsInCurrentTeam']=='Y']

FixturesElementsAll3FilteredDef=FixturesElementsAll3FilteredDef[FixturesElementsAll3FilteredDef['element_type_x']<3]

FixturesElementsAll3FilteredDefslim = FixturesElementsAll3FilteredDef[['web_name','event','AttackOpp_pergame','DefenceOpp_pergame']]
#FixturesElementsAll3Filteredslim['event'].astype(int)
FixturesElementsAll3FilteredDefslim['DefenceOpp_pergame'].astype(float)
FixturesElementsAll3FilteredDefslim['AttackOpp_pergame'].astype(float)
FixturesElementsAll3FilteredDefslim.sort_values(by='event', ascending=True)

FixturesElementsAll3FilteredDefslim=FixturesElementsAll3FilteredDefslim.drop_duplicates(subset=['event','web_name'])

st.write("Current Team Defence Defensive Rating - Lower is Better")

st.bar_chart(data=FixturesElementsAll3FilteredDefslim, x="event", y="DefenceOpp_pergame",color="web_name", horizontal=False, stack=False)

st.write("Current Team Defence Attacking Rating")

st.bar_chart(data=FixturesElementsAll3FilteredDefslim, x="event", y="AttackOpp_pergame",color="web_name", horizontal=False, stack=False)


##### Returns Output 2 FixturesElementsAllGrouped for previous 4 weeks

FixturesElementsAllGrouped = FixturesElementsAll2[FixturesElementsAll2['event']>UpcomingWeek-NoWeeks-1]

FixturesElementsAllGrouped = FixturesElementsAllGrouped[['playerid','web_name','element_type','team','now_cost','expected_goals.gw','expected_assists.gw','expected_goal_involvements.gw','expected_goals_conceded.gw','XGI_Weighted','XGC_Weighted','AttackOpp_pergame','DefenceOpp_pergame']]

FixturesElementsAllGrouped = FixturesElementsAllGrouped.groupby(['playerid','web_name','element_type','now_cost','team'], as_index=False)[['expected_goals.gw','expected_assists.gw','expected_goal_involvements.gw','expected_goals_conceded.gw','XGI_Weighted','XGC_Weighted','AttackOpp_pergame','DefenceOpp_pergame']].mean(numeric_only=True)

FixturesElementsAllGrouped.to_csv('FixturesElementsAllGrouped.csv')

FixturesElementsAllGroupedwithTS = pd.merge(FixturesElementsAllGrouped, TeamSelection,left_on='playerid', right_on='element', how='outer')

FixturesElementsAllGroupedwithTS['IsInCurrentTeam'] = np.where(
    FixturesElementsAllGroupedwithTS['is_captain'].notnull(),
    'Y',
    'N'
)

##Import the packages to carry out the optimisation problem

import requests
#from sklearn.preprocessing import MinMaxScaler
import warnings
from pulp import *
Small_Elements_df = FixturesElementsAllGroupedwithTS

def AttackOptimisationproblem(Source,AttSpend,FreeTransfers):
    Attack_Elements_df = Source.dropna(subset=['AttackOpp_pergame'])
    Attack_Elements_df['AttackOpp_pergame'].astype(float)
    Attack_Elements_df=Attack_Elements_df[Attack_Elements_df['element_type_x']!=1]
    Attack_Elements_df=Attack_Elements_df[Attack_Elements_df['element_type_x']!=2]
    ## Attack Variables
    Players = list(Attack_Elements_df['web_name'])
    Cost = dict(zip(Players, Attack_Elements_df['now_cost']))
    Score = dict(zip(Players, Attack_Elements_df['AttackOpp_pergame'].astype(float)))
    positions = dict(zip(Players, Attack_Elements_df['element_type_x']))
    CurrentSelection = dict(zip(Players, Attack_Elements_df['IsInCurrentTeam']))
    
    player_vars = LpVariable.dicts("Player", Players, lowBound=0, upBound=1, cat='Integer')

    ##Attack Constraints and problem solve

    total_score = LpProblem("Fantasy_Points_Problem", LpMaximize)
    total_score += lpSum([Score[i] * player_vars[i] for i in player_vars])
    total_score += lpSum([Cost[i] * player_vars[i] for i in player_vars]) <= AttSpend

    MF = [p for p in positions.keys() if positions[p] == 3]
    ST = [p for p in positions.keys() if positions[p] == 4]
    CS = [p for p in positions.keys() if CurrentSelection[p] == 'N']
    total_score += lpSum([player_vars[i] for i in MF]) == 5
    total_score += lpSum([player_vars[i] for i in ST]) == 3
    total_score += lpSum([player_vars[i] for i in CS]) == 8-FreeTransfers

    total_score.solve()

    for v in total_score.variables():
        if v.varValue > 0:
            print(v.name)

def DefenceOptimisationproblem(Source,DefSpend,FreeTransfers):
    Def_Elements_df=Source.dropna(subset=['DefenceOpp_pergame'])
    Def_Elements_df['DefenceOpp_pergame'].astype(float)
    Def_Elements_df=Def_Elements_df[Def_Elements_df['element_type_x']!=3]
    Def_Elements_df=Def_Elements_df[Def_Elements_df['element_type_x']!=4]
    ## Defence Variables
    DefPlayers = list(Def_Elements_df['web_name'])
    DefCost = dict(zip(DefPlayers, Def_Elements_df['now_cost']))
    DefScore = dict(zip(DefPlayers, Def_Elements_df['DefenceOpp_pergame'].astype(float)))
    Defpositions = dict(zip(DefPlayers, Def_Elements_df['element_type_x']))
    DefCurrentSelection = dict(zip(DefPlayers, Def_Elements_df['IsInCurrentTeam']))

    player_vars_def = LpVariable.dicts("DefPlayers", DefPlayers, lowBound=0, upBound=1, cat='Integer')

##Defence Constraints and problem solve

    total_score_def = LpProblem("Fantasy_Points_Problem_Defence", LpMinimize)
    total_score_def += lpSum([DefScore[i] * player_vars_def[i] for i in player_vars_def])
    total_score_def += lpSum([DefCost[i] * player_vars_def[i] for i in player_vars_def]) <= DefSpend

    GK = [p for p in Defpositions.keys() if Defpositions[p] == 1]
    DF = [p for p in Defpositions.keys() if Defpositions[p] == 2]
    #CS = [p for p in DefCurrentSelection.keys() if DefCurrentSelection[p] == 'N']
    total_score_def += lpSum([player_vars_def[i] for i in GK]) == 2
    total_score_def += lpSum([player_vars_def[i] for i in DF]) == 5
    #total_score_def += lpSum([player_vars_def[i] for i in CS]) == 7-FreeTransfers

    total_score_def.solve()

    for v in total_score_def.variables():
        if v.varValue > 0:
            print(v.name)

AttackOptimisationproblem(Small_Elements_df,700,1)
DefenceOptimisationproblem(Small_Elements_df,300,1)

st.write(AttackOptimisationproblem(Small_Elements_df,700,1))

st.write(DefenceOptimisationproblem(Small_Elements_df,300,1))

