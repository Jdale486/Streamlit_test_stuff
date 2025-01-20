##import packages

import requests
import json
import pandas as pd
import numpy as np
from decimal import Decimal
import itertools as it
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

number = st.number_input("Enter Team ID", step=1, placeholder = 1769272)
st.write("The current Team ID is ", number)

#NoPlayersShown = st.number_input("Enter No. of Players to View", step=1,placeholder=10)
#st.write("No. of Players Shown:", NoPlayersShown)


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

NoTrailingWeeks = 4

NoForecastWeeks = 5

#TeamID = TeamIDSel.astype(float)

TeamID = number

#NoPlayersShownInt = NoPlayersShown.astype(float)

#NoPlayersShownInt = float(NoPlayersShown)

# TeamID = 1769272

NoPlayersShownInt = 20

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
    "Teams.name2": [
        "Bournemouth", "Crystal Palace", "Nott'm Forest", "Wolves", "Brentford", "Fulham",
        "Everton", "Aston Villa", "Newcastle", "West Ham", "Brighton", "Man Utd", "Chelsea",
        "Liverpool", "Spurs", "Arsenal", "Man City", "Ipswich", "Leicester", "Southampton"
    ]
    ,
    "Teams.name": [
        "Bour", "CryP", "For't", "Wolv", "Bfrd", "Ful",
        "Ev", "Villa", "Newc", "WHam", "Bri", "ManU", "Chel",
        "LPool", "Spurs", "Ars", "ManC", "Ips", "Leic", "Soton"
    ]
        ,
    "Teams.name1": [
        "b", "c", "F", "W", "B", "F",
        "E", "V", "N", "W", "B", "M", "Ch",
        "L", "S", "A", "M", "I", "L", "S"
    ]
}

TeamWeightings = pd.DataFrame(TeamWeightings)

Elements_Dim1['OppTeam'] = Elements_Dim1.apply(lambda row: row['team_h'] if row['team_h'] == row['team'] else row['team_a'], axis=1)

Elements_Dim2 = pd.merge(Elements_Dim1, TeamWeightings,left_on='OppTeam', right_on='Teams.id', how='outer')

Elements_Dim2['XGC_Weighted'] = Elements_Dim2['Weighting'].astype(float)*Elements_Dim2['expected_goals_conceded.gw'].astype(float)

Elements_Dim2['XGI_Weighted'] = Elements_Dim2['expected_goal_involvements.gw'].astype(float)/Elements_Dim2['Weighting'].astype(float)

###Filtered to only include the previous 4 weeks' data before it is grouped to return total sum for each team 

Elements_Dim2Filtered = Elements_Dim2[Elements_Dim2['event']>PrevWeek-NoTrailingWeeks]

Elements_Dim2Grouped1 = Elements_Dim2Filtered.groupby(['OppTeam','Teams.name', 'event']).agg(
    XGI_Weighted=('XGI_Weighted', 'sum'),
    XGC_Weighted=('XGC_Weighted', 'max'),
    Minutes=('minutes_x', 'sum')
).reset_index()



##Average of the above taken for each team
Elements_Dim2Grouped2 = Elements_Dim2Grouped1.groupby(['OppTeam','Teams.name']).agg(
    XGI_Weighted=('XGI_Weighted', 'mean'),
    XGC_Weighted=('XGC_Weighted', 'mean'),
    Minutes=('Minutes', 'mean')
).reset_index()



Elements_Dim2Grouped2.head()

Elements_Dim2Grouped2.to_csv('TeamProfiles.csv')

##Previous 4 Gameweeks this dataframe is returning the averages for the selected players
Elements_Dim1 = Elements_Dim1[Elements_Dim1['event']<UpcomingWeek]

GWElementsDFLast4Weeks = Elements_Dim1[Elements_Dim1['event']>UpcomingWeek-NoTrailingWeeks-1]

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

url = f"https://fantasy.premierleague.com/api/entry/{TeamID}/event/{PrevWeek}/picks/"
response = requests.get(url)
json_data = response.json()

TeamSelection = json_data['picks']

TeamSelection = pd.json_normalize(TeamSelection)

TeamSelection1 = TeamSelection[['element','is_captain']]

TeamSelection1.head()

######### DataframeFiltering



######### Midfielders

##### Returns Output 1 FixturesElementsAll
FixturesElementsAll1 = pd.merge(FixturesConcat, Elements_Dim2Grouped2,left_on='OppTeam2', right_on='OppTeam', how='outer')

FixturesElementsAll2 = pd.merge(grouped_dfLast4Weeks, FixturesElementsAll1,left_on='team', right_on='TeamID', how='outer')

FixturesElementsAll3 =  pd.merge(TeamSelection1, FixturesElementsAll2,left_on='element', right_on='playerid', how='outer')

FixturesElementsAll3['InCurrentTeam']= np.where(
    FixturesElementsAll3['element'].notnull(),
    1,
    0
)

FixturesElementsAll3['AttackOpp_pergame'] = FixturesElementsAll3['XGC_Weighted']*FixturesElementsAll3['expected_goal_involvements.gw']

FixturesElementsAll3['DefenceOpp_pergame'] = FixturesElementsAll3['XGI_Weighted']*FixturesElementsAll3['expected_goals_conceded.gw']

FixturesElementsAll3.to_csv('FixturesElementsAll.csv')

FixturesElementsAll3['xGIAvg_Player'] = FixturesElementsAll3['expected_goal_involvements.gw'].apply(lambda x: format(x, ".2f")).astype(str) + '-' + FixturesElementsAll3['web_name']


FixturesElementsAll2Att=FixturesElementsAll3[FixturesElementsAll3['element_type']==3]

# Calculate total AttackOpp_pergame for each player
FixturesElementsAll2Att = FixturesElementsAll2Att[FixturesElementsAll2Att['event']<UpcomingWeek+6]
FixturesElementsAll2_totalsAtt = FixturesElementsAll2Att.groupby('web_name')['AttackOpp_pergame'].sum().reset_index()

# Sort players by total AttackOpp_pergame and select top 20
top_players = FixturesElementsAll2_totalsAtt.sort_values(by='AttackOpp_pergame', ascending=False).head(NoPlayersShownInt)

# Filter original DataFrame to include only top 20 players
FixturesElementsAll2AttFiltered = FixturesElementsAll2Att[FixturesElementsAll3['web_name'].isin(top_players['web_name'])]


# Pivot the DataFrame using pivot_table with an aggregation function
heatmap_data = FixturesElementsAll2AttFiltered.pivot_table(
    index='xGIAvg_Player',
    columns='event',
    values='XGC_Weighted',
    aggfunc='mean'
)
text_data = FixturesElementsAll2AttFiltered.pivot_table(
    index='xGIAvg_Player',
    columns='event',
    values='Teams.name',
    aggfunc=lambda x: ' | '.join(x.astype(str))
)
current_team_data = FixturesElementsAll2AttFiltered.set_index('xGIAvg_Player')['InCurrentTeam'].to_dict()

# Prepare custom tick labels for y-axis
yaxis_tickvals = list(heatmap_data.index)
yaxis_ticktext = [f'<b style="color:blue;">{player}</b>' if current_team_data[player] > 0 else player for player in yaxis_tickvals]

# Create the heatmap
fig = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=yaxis_tickvals,
    text=text_data.values,
    texttemplate="%{text}",
    textfont={"size":8},
    colorscale='reds'
))

# Update layout to leave more space for the y-axis
fig.update_layout(
    margin=dict(l=200, r=50, t=50, b=50),  # Increase left margin to make room for y-axis labels
    yaxis=dict(
        tickvals=yaxis_tickvals,
        ticktext=yaxis_ticktext
    )
)

# Display the figure
#fig.show()
st.write("Midfielders")


st.plotly_chart(fig, use_container_width=True)



#############Attackers



##### Returns Output 1 FixturesElementsAll
FixturesElementsAll1 = pd.merge(FixturesConcat, Elements_Dim2Grouped2,left_on='OppTeam2', right_on='OppTeam', how='outer')

FixturesElementsAll2 = pd.merge(grouped_dfLast4Weeks, FixturesElementsAll1,left_on='team', right_on='TeamID', how='outer')

FixturesElementsAll3 =  pd.merge(TeamSelection1, FixturesElementsAll2,left_on='element', right_on='playerid', how='outer')

FixturesElementsAll3['InCurrentTeam']= np.where(
    FixturesElementsAll3['element'].notnull(),
    1,
    0
)

FixturesElementsAll3['AttackOpp_pergame'] = FixturesElementsAll3['XGC_Weighted']*FixturesElementsAll3['expected_goal_involvements.gw']

FixturesElementsAll3['DefenceOpp_pergame'] = FixturesElementsAll3['XGI_Weighted']*FixturesElementsAll3['expected_goals_conceded.gw']

FixturesElementsAll3.to_csv('FixturesElementsAll.csv')

FixturesElementsAll3['xGIAvg_Player'] = FixturesElementsAll3['expected_goal_involvements.gw'].apply(lambda x: format(x, ".2f")).astype(str) + '-' + FixturesElementsAll3['web_name']


FixturesElementsAll2Att=FixturesElementsAll3[FixturesElementsAll3['element_type']==4]

# Calculate total AttackOpp_pergame for each player
FixturesElementsAll2Att = FixturesElementsAll2Att[FixturesElementsAll2Att['event']<UpcomingWeek+6]
FixturesElementsAll2_totalsAtt = FixturesElementsAll2Att.groupby('web_name')['AttackOpp_pergame'].sum().reset_index()

# Sort players by total AttackOpp_pergame and select top 20
top_players = FixturesElementsAll2_totalsAtt.sort_values(by='AttackOpp_pergame', ascending=False).head(NoPlayersShownInt)

# Filter original DataFrame to include only top 20 players
FixturesElementsAll2AttFiltered = FixturesElementsAll2Att[FixturesElementsAll3['web_name'].isin(top_players['web_name'])]


# Pivot the DataFrame using pivot_table with an aggregation function
heatmap_data = FixturesElementsAll2AttFiltered.pivot_table(
    index='xGIAvg_Player',
    columns='event',
    values='XGC_Weighted',
    aggfunc='mean'
)
text_data = FixturesElementsAll2AttFiltered.pivot_table(
    index='xGIAvg_Player',
    columns='event',
    values='Teams.name',
    aggfunc=lambda x: ' | '.join(x.astype(str))
)
current_team_data = FixturesElementsAll2AttFiltered.set_index('xGIAvg_Player')['InCurrentTeam'].to_dict()

# Prepare custom tick labels for y-axis
yaxis_tickvals = list(heatmap_data.index)
yaxis_ticktext = [f'<b style="color:blue;">{player}</b>' if current_team_data[player] > 0 else player for player in yaxis_tickvals]

# Create the heatmap
fig1 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=yaxis_tickvals,
    text=text_data.values,
    texttemplate="%{text}",
    textfont={"size":8},
    colorscale='reds'
))

# Update layout to leave more space for the y-axis
fig1.update_layout(
    margin=dict(l=200, r=50, t=50, b=50),  # Increase left margin to make room for y-axis labels
    yaxis=dict(
        tickvals=yaxis_tickvals,
        ticktext=yaxis_ticktext
    )
)

# Display the figure
#fig.show()
st.write("Attackers")


st.plotly_chart(fig1, use_container_width=True)