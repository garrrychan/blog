Title: Do NBA teams with greater team salaries win more? 
Date: 2019-03-30 3:00
Tags: python
Slug: nba_team_salary

FiveThirtyEight is one of my favourite places to read thoughtful analysis, and see data visualizations. I was inspired by one of their posts on ['how your favourite baseball team blows its money'](https://fivethirtyeight.com/features/how-your-favorite-baseball-team-blows-its-money/). As an avid basketball fan, I recreated this graphic with NBA data. 

This was incredible practice with Matplotlib. Thank you to Amy Gordon for her [Github](https://github.com/ag2816/) for the style guide.

### Background

<img src="images/nba_team_salary_40_0.svg" alt="nba_salary" class="img-responsive">

This chart illustrates the relationship between team spending and win percentage for each basketball season from 2001 - 2015. Essentially, the steeper the curve, the better the team was maximizing value. 

* Each regular season is one dot in the figure
* Salaries are inflation adjusted, and standardized.
* The coloured line in each chart represents the linear regression fit for a team
* The grey line represents the linear regression fit for the league in aggregate

More money, more wins. There is a rough trend suggesting that higher team salaries lead to a greater winning percentage. This relationship is particularly strong for the Boston Celtics and Golden State Warriors.

On the other hand, this relationship is flat for others. For example, irrespective of team salary, the San Antonio Spurs is a winning organization. They are better at managing its roster and payroll versus others. They pride themselves with developing talent organically with scouting, coaching, and player development. Rather than overpaying Superstar players in the free agency market, they sign undervalued players (players who contribute more to win share, relatively their salary would suggest). This is similar to the thinking in MoneyBall, and has been widely studied in sports. Coupled with the fact that there is usually a salary cap limit in professional sports, this becomes a very interesting dynamic. 

Worst out of all the teams, New York Knicks tend to overpay players, which don't yield any marginal wins on the court and underperform relative to expectations.

---

#### Data

Game Data:
https://www.kaggle.com/fivethirtyeight/fivethirtyeight-nba-elo-dataset

Salary data (exported to CSV):
https://hoopshype.com/salaries/


For those interested, keep reading for data wrangling and snippets of code.
<br>

---
### 1. Load & prepare data



```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# blog html visualization 
from IPython.display import HTML

# plt.rcParams['figure.dpi'] = 50
# %config InlineBackend.figure_format = 'retina'
%config InlineBackend.figure_format = 'svg'
```

Salary data


```python
raw=pd.read_csv('./data/nba_team_salary.csv')
```


```python
# reorganize columns in chronological order
salary = pd.concat([raw["team"], raw[raw.columns[:0:-1]]], axis=1)

# change NaN to zero for no salary expansion teams: Charlotte 2003, 2004; New Orleans 2001, 2002
salary.fillna("$0", inplace=True)

# rename city to mascot name to merge later with games dataset
teams = {"team":["Hawks","Celtics","Nets","Hornets","Bulls","Cavaliers","Mavericks","Nuggets",
                 "Pistons","Warriors","Rockets","Pacers","Clippers","Lakers","Grizzlies","Heat",
                 "Bucks","Timberwolves","Pelicans","Knicks","Thunder","Magic","Sixers","Suns",
                 "Trailblazers","Kings","Spurs","Raptors","Jazz","Wizards"]}
           
salary = pd.concat([pd.DataFrame(teams),salary.iloc[:,1:16]],axis=1)

# transform wide to long data for plotting
# 450 columns: 30 teams x 15 years
long_salary = pd.melt(salary, id_vars='team', var_name='year', value_name='salary')
long_salary.head()

# year into int
long_salary["year"] = long_salary["year"].apply(lambda x: int(x))

# salary into int
long_salary["salary"] = long_salary["salary"].apply(lambda x: int(x.replace("$","").replace(",","")))
```

---
To make the data more comparable year over year, let's standardize the salaries. This function calculates how many standard deviations a team's salary is from the league average for a given year and returns that value. 


```python
def z_salary(df,year,salary):
    year_mean = df[(df['year']==year)]['salary'].mean()
    year_std = df[(df['year']==year)]['salary'].std()
    return (salary - year_mean)/year_std
```


```python
long_salary['z_salary'] = long_salary.apply(lambda row: z_salary(long_salary, row["year"], row["salary"]), axis=1)
HTML(long_salary.head().to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>year</th>
      <th>salary</th>
      <th>z_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hawks</td>
      <td>2001</td>
      <td>57438766</td>
      <td>-0.689421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Celtics</td>
      <td>2001</td>
      <td>75244004</td>
      <td>0.086161</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nets</td>
      <td>2001</td>
      <td>100821283</td>
      <td>1.200288</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hornets</td>
      <td>2001</td>
      <td>67881439</td>
      <td>-0.234546</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bulls</td>
      <td>2001</td>
      <td>43399261</td>
      <td>-1.300970</td>
    </tr>
  </tbody>
</table>




```python
# export to csv 
long_salary.to_csv("./data/nba_team_salary_final.csv")
```

---
Games data

From this NBA Elo dataset from Kaggle on FiveThirtyEight, I'll need to tally the following for each season. 
* G: Games play
* W: Games won
* L: Games lost

In the future, I'll be web scraping, which will make this easier. 


```python
games = pd.read_csv('./data/nba_elo.csv')

# I'll only look at the regular season to keep n consistent across teams
games = games[games["is_playoffs"] == 0]

# This dataset has a lot of extraneous information, so I'll focus on the columns of interest. 
games = games[["year_id","date_game","team_id","fran_id","opp_id","opp_fran","game_result"]]

# This dataset is from 1947-2019, so I'll filter for years 2001 to 2015 to match our Salary data
games = games[(games["year_id"] >= 2001) & (games["year_id"] <= 2015)]

HTML(games.head().to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year_id</th>
      <th>date_game</th>
      <th>team_id</th>
      <th>fran_id</th>
      <th>opp_id</th>
      <th>opp_fran</th>
      <th>game_result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87750</th>
      <td>2001</td>
      <td>10/31/2000</td>
      <td>ATL</td>
      <td>Hawks</td>
      <td>CHH</td>
      <td>Pelicans</td>
      <td>L</td>
    </tr>
    <tr>
      <th>87751</th>
      <td>2001</td>
      <td>10/31/2000</td>
      <td>CHH</td>
      <td>Pelicans</td>
      <td>ATL</td>
      <td>Hawks</td>
      <td>W</td>
    </tr>
    <tr>
      <th>87752</th>
      <td>2001</td>
      <td>10/31/2000</td>
      <td>CHI</td>
      <td>Bulls</td>
      <td>SAC</td>
      <td>Kings</td>
      <td>L</td>
    </tr>
    <tr>
      <th>87753</th>
      <td>2001</td>
      <td>10/31/2000</td>
      <td>SAC</td>
      <td>Kings</td>
      <td>CHI</td>
      <td>Bulls</td>
      <td>W</td>
    </tr>
    <tr>
      <th>87754</th>
      <td>2001</td>
      <td>10/31/2000</td>
      <td>DAL</td>
      <td>Mavericks</td>
      <td>MIL</td>
      <td>Bucks</td>
      <td>W</td>
    </tr>
  </tbody>
</table>




```python
# reset record to beginning with list comphrension
record = {team:{"G":0,"W":0,"L":0} for team in teams}
```


```python
# Each game shows up for the home and away team, so I'll only need to run control flow for when they lose
# Repeat for each year

def create_record(year):
    for i, row in games[games["year_id"]==year].iterrows():
        if row.game_result == "L":
            record[row.fran_id]["G"] += 1
            record[row.opp_fran]["G"] += 1
            record[row.fran_id]["L"] += 1
            record[row.opp_fran]["W"] += 1
    return record
```


```python
# set up list of each team and each year
teams = sorted(list(games.fran_id.unique()))
all_years = list(games.year_id.unique())
```


```python
# now repeat for all the years in a dictionary object
seasons = {}
for year in all_years:
    record = {team:{"G":0,"W":0,"L":0} for team in teams}
    seasons[year] = create_record(year)
```


```python
create_record(2001)
```




    {'Bucks': {'G': 164, 'W': 93, 'L': 71},
     'Bulls': {'G': 164, 'W': 65, 'L': 99},
     'Cavaliers': {'G': 164, 'W': 83, 'L': 81},
     'Celtics': {'G': 164, 'W': 76, 'L': 88},
     'Clippers': {'G': 164, 'W': 87, 'L': 77},
     'Grizzlies': {'G': 164, 'W': 78, 'L': 86},
     'Hawks': {'G': 164, 'W': 85, 'L': 79},
     'Heat': {'G': 164, 'W': 87, 'L': 77},
     'Hornets': {'G': 82, 'W': 33, 'L': 49},
     'Jazz': {'G': 164, 'W': 91, 'L': 73},
     'Kings': {'G': 164, 'W': 84, 'L': 80},
     'Knicks': {'G': 164, 'W': 65, 'L': 99},
     'Lakers': {'G': 164, 'W': 77, 'L': 87},
     'Magic': {'G': 164, 'W': 68, 'L': 96},
     'Mavericks': {'G': 164, 'W': 103, 'L': 61},
     'Nets': {'G': 164, 'W': 64, 'L': 100},
     'Nuggets': {'G': 164, 'W': 70, 'L': 94},
     'Pacers': {'G': 164, 'W': 79, 'L': 85},
     'Pelicans': {'G': 164, 'W': 91, 'L': 73},
     'Pistons': {'G': 164, 'W': 64, 'L': 100},
     'Raptors': {'G': 164, 'W': 96, 'L': 68},
     'Rockets': {'G': 164, 'W': 101, 'L': 63},
     'Sixers': {'G': 164, 'W': 74, 'L': 90},
     'Spurs': {'G': 164, 'W': 113, 'L': 51},
     'Suns': {'G': 164, 'W': 90, 'L': 74},
     'Thunder': {'G': 164, 'W': 89, 'L': 75},
     'Timberwolves': {'G': 164, 'W': 63, 'L': 101},
     'Trailblazers': {'G': 164, 'W': 101, 'L': 63},
     'Warriors': {'G': 164, 'W': 84, 'L': 80},
     'Wizards': {'G': 164, 'W': 65, 'L': 99}}




```python
# rename index column to team
df_seasons = pd.DataFrame(seasons).reset_index().rename(columns={'index':'team'})

# wide to long data for plotting
long_seasons = pd.melt(df_seasons, id_vars='team', var_name='year', value_name='seasons')

# Break out the G,W,L into columns
items = list(long_seasons["seasons"])

long_seasons["G"] = pd.DataFrame([item["G"] for item in items])
long_seasons["L"] = pd.DataFrame([item["L"] for item in items])
long_seasons["W"] = pd.DataFrame([item["W"] for item in items])

# Calculate winning % for each team per year, and add as a new column
long_seasons['win_%'] = round(long_seasons['W']/long_seasons['G'],2)

# drop seasons, redundant column with G, L, W
long_seasons = long_seasons.drop(columns=['seasons'])

# make year into int so it can be merged later
long_seasons["year"] = long_seasons["year"].apply(lambda x: int(x))
```


```python
HTML(long_seasons.head().to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
      <th>year</th>
      <th>G</th>
      <th>L</th>
      <th>W</th>
      <th>win_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bucks</td>
      <td>2001</td>
      <td>82</td>
      <td>30</td>
      <td>52</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bulls</td>
      <td>2001</td>
      <td>82</td>
      <td>67</td>
      <td>15</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cavaliers</td>
      <td>2001</td>
      <td>82</td>
      <td>52</td>
      <td>30</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Celtics</td>
      <td>2001</td>
      <td>82</td>
      <td>46</td>
      <td>36</td>
      <td>0.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Clippers</td>
      <td>2001</td>
      <td>82</td>
      <td>51</td>
      <td>31</td>
      <td>0.38</td>
    </tr>
  </tbody>
</table>



Now, I'll merge the two datasets (long_salary, long_seasons) together into one dataframe for scatterplots


```python
# merge datasets together
final = long_salary.merge(long_seasons, on=["team","year"])
final

# add divisions data
divisions_dict = {"Atlantic": ["Celtics","Nets","Sixers","Raptors","Knicks"],
             "Central": ["Bulls","Cavaliers","Pistons","Bucks","Pacers"],
             "Southeast": ["Hawks","Hornets","Heat","Magic","Wizards"],
             "Northwest": ["Timberwolves","Nuggets","Trailblazers","Thunder","Jazz"],
             "Pacific" : ["Warriors","Clippers","Lakers","Kings","Suns"],
             "Southwest" : ["Mavericks", "Rockets","Grizzlies","Pelicans","Spurs"]}

divisions = ["Atlantic","Central","Southeast","Northwest","Pacific","Southwest"]

def team_to_division(team):
    for division in divisions:
        if team in divisions_dict[division]:
            return division

final["division"] = final["team"].apply(lambda x: team_to_division(x))

# add conference data
conferences = ["East","West"]
conferences_dict = {"East": ["Celtics","Nets","Sixers","Raptors","Knicks","Bulls","Cavaliers","Pistons","Bucks","Pacers","Hawks","Hornets","Heat","Magic","Wizards"],
                   "West": ["Timberwolves","Nuggets","Trailblazers","Thunder","Jazz","Warriors","Clippers","Lakers","Kings","Suns","Mavericks", "Rockets","Grizzlies","Pelicans","Spurs"]}

def division_to_conference(division):
    for conference in conferences:
        if division in conferences_dict[conference]:
            return conference
        
final["conference"] = final["team"].apply(lambda x: division_to_conference(x))
```


```python
# Remove early Charlotte Hornet years, missing data;
# they relocated to New Orleans, and were renamed as the Pelicans
final[final["team"] == 'Hornets'].loc[[3,33,63,93]]
final = final.drop([3,33,63,93])
```


```python
final.to_csv("./data/final.csv")
```

---

### 2. Create plot for one team

* Use matplotlib, a Python 2D plotting library
* Create a scatterplot for the Raptors


```python
raptors = final[final['team']== 'Raptors']
```


```python
# change way we invoke plot so can manipulate the axis
fig, ax = plt.subplots(figsize=(3, 3))

#ax.scatter(x, y)
ax.scatter(x=raptors['z_salary'], y=raptors['win_%'],alpha=0.5,c="#CE1141")
plt.title('Toronto Raptors', position = (0,1), ha = 'left', fontsize=12)
plt.xlabel("Standardized Salaries", position = (0,0), ha = 'left', color = 'grey')
plt.ylabel('Win %', position = (0, 1), ha = 'right', color = 'grey')

# add linear regression line
X = pd.DataFrame(final[final["team"]=="Raptors"]["z_salary"])
Y = pd.DataFrame(raptors["win_%"])
linreg = linear_model.LinearRegression()
model = linreg.fit(X, Y)
predictions = model.predict(X)

plt.plot(X, predictions, color='#CE1141',linewidth=2)

# remove the top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# add in gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# create cross hairs 
plt.hlines(0.5,-3, 3,alpha=0.5) 
plt.vlines(0,0,1,alpha=0.5)

# reduce ticks on x axis
plt.xticks([-2,-1,0,1,2])

plt.show()
```


![svg](images/nba_team_salary_30_0.svg)


Compare this with the plot for the league 


```python
# change way we invoke plot so can manipulate the axis
fig, ax = plt.subplots(figsize=(3, 3))

#ax.scatter(x, y)
scatter_league = (ax.scatter(x=final['z_salary'], y=final['win_%'],alpha=0.5,c="#000000"),
                  plt.title('NBA League', position = (0,1), ha = 'left', fontsize=12),
                  plt.xlabel("Standardized Salaries", position = (0,0), ha = 'left', color = 'grey'),
                  plt.ylabel('Win %', position = (0, 1), ha = 'right', color = 'grey'))

# add linear regression line
X = pd.DataFrame(final["z_salary"])
Y_league = pd.DataFrame(final["win_%"])
linreg = linear_model.LinearRegression()
model = linreg.fit(X, Y_league)
predictions_league = model.predict(X)

# remove the top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# add in gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# create cross hairs 
plt.hlines(0.5,-3, 3,alpha=0.5) 
plt.vlines(0,0,1,alpha=0.5)

# reduce ticks on x axis
plt.xticks([-2,-1,0,1,2])

league_plot = plt.plot(X, predictions_league, color='#000000',linewidth=2,alpha=0.25)
plt.show()
```


![svg](images/nba_team_salary_32_0.svg)


---
### 3. Generate plots for all the teams in a division, in a row


```python
# Create a dictionary with the team names, and team plot colour. I'll do this with Atlantic, 
# then loop through each division on each row

atlantic = {"Celtics": "#007A33",
            "Nets": "#000000",
            "Sixers": "#006BB6",
            "Raptors": "#CE1141",
            "Knicks": "#F58426"}
```


```python
def plot_team_row(team_df, ax,team_color='#00000'):

    # linear regression
    X = pd.DataFrame(team_df["z_salary"])
    Y = pd.DataFrame(team_df["win_%"])
    linreg = linear_model.LinearRegression()
    model = linreg.fit(X, Y)
    predictions = model.predict(X)
    ax.plot(X, predictions, color=team_color,alpha=0.8,linewidth=2)    
    
    # plot league average
    ax.plot(pd.DataFrame(final['z_salary']), predictions_league, color='#C4CED4',alpha=0.5,linewidth=2);

    # scatterplot
    ax.scatter(x=team_df['z_salary'], y=team_df['win_%'],alpha=0.5,c=team_color,s=10)
    
    # title for each team subplot
    ax.set_title(f"{team_df['team'].values[0]}",fontsize=10)
 
    # create cross hairs
    ax.hlines(0.5,-3, 3,alpha=0.5,linewidth=2)
    ax.vlines(0,.10,.90,alpha=0.5,linewidth=2)

    # ticks
    plt.xticks([-2,0,2])

    # remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # add in gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
```

* Outside of the loop, instantiate the plot with 1 row x 5 columns each of the teams in the division
* The `ax` object is an array with each array defining one subplot
* These arrays can be passed into to the previous `plot_team_row()` function


```python
fig, ax = plt.subplots(nrows=1, ncols=5,figsize=(6,2),sharex=True, sharey=True)

# Division title
plt.figtext(0.48,1, "Atlantic", fontsize=10, color="grey")

# Spacing for subplots
fig.subplots_adjust(hspace = 0.5, wspace=.001, top=0.825)

# pos determines which axes object to pass into the function
#there are 5 pos
col=0 
for team, colour in atlantic.items():
    plot_team_row(final[final['team']==team], ax[col],colour)
    col=col+1 
    
# force label to appear on first plot rather than the last plot
ax[0].set_xlabel('Standardized Salaries', fontsize=10)
ax[0].set_ylabel('Win %', fontsize=10)
ax[0].tick_params(labelsize=8)
ax[1].tick_params(labelsize=8)
ax[2].tick_params(labelsize=8)
ax[3].tick_params(labelsize=8)
ax[4].tick_params(labelsize=8)

plt.show()
```


![svg](images/nba_team_salary_37_0.svg)


---
### 4. Create the full chart for the entire NBA


```python
# team colours source: https://teamcolorcodes.com/

atlantic = {"Celtics": "#007A33",
            "Nets": "#000000",
            "Sixers": "#006BB6",
            "Raptors": "#CE1141",
            "Knicks": "#F58426"}

central = {"Bulls": "#CE1141",
            "Cavaliers": "#6F263D",
            "Pistons": "#C8102E",
            "Bucks": "#00471B",
            "Pacers": "#FDBB30"}

southeast = {"Hawks": "#E03A3E",
             "Hornets":"#00788C",
             "Heat": "#98002E",
             "Magic": "#0077C0",
             "Wizards": "#002B5C"}

northwest = {"Timberwolves": "#236192",
            "Nuggets": "#FEC524",
            "Trailblazers": "#E03A3E",
            "Thunder": "#007AC1",
            "Jazz": "#00471B"}

pacific = {"Warriors": "#FDB927",
           "Lakers": "#552583",
           "Clippers": "#C8102E",
           "Kings": "#5A2D81",
           "Suns": "#E56020"}

    
southwest= {"Mavericks": "#00538C",
            "Rockets": "#CE1141",
            "Grizzlies": "#5D76A9",
            "Pelicans": "#85714D",
            "Spurs": "#C4CED4"}

nba_league = [atlantic, central, southeast, northwest, pacific, southwest]
```


```python
# 6 rows x 5 columns for all 30 teams
fig, ax = plt.subplots(nrows=6, ncols=5,figsize=(6,6),sharex=True, sharey=True)

# plot title
fig.suptitle('More Money, More Wins', x=0.55, y=1,color = 'black', fontsize=12) 

# Subplots spacing
fig.subplots_adjust(hspace = .5, wspace=.001, top=0.925)

# ax[count][pos]: the axes object contains a numpy array
# count indicates the row
# pos indicates the column
# loop through each division, each team
row = 0 
for division in nba_league:
    col=0 
    for team,color in division.items():
        plot_team_row(final[final['team']==team], ax[row][col],color)
        col=col+1
    row += 1
    
# labels
ax[5][2].set_xlabel('Standardized Salary', fontsize=8)
    
ax[0][0].set_ylabel('Win %', fontsize=8)

# ticks
for i in list(range(0,5)):
    ax[5][i].tick_params(labelsize=8)

for i in list(range(0,5)):
    ax[i][0].tick_params(labelsize=8)

plt.figtext(1,0.875, "Atlantic", fontsize=10, color="grey")
plt.figtext(1,0.725, "Central", fontsize=10, color="grey")
plt.figtext(1,0.575, "Southeast", fontsize=10, color="grey")
plt.figtext(1,0.425, "Northwest", fontsize=10, color="grey")
plt.figtext(1,0.275, "Pacific", fontsize=10, color="grey")
plt.figtext(1,0.125, "Southwest", fontsize=10, color="grey") 
plt.show()
```


![svg](images/nba_team_salary_40_0.svg)



```python
# plt.savefig('nba_team_salary.pdf')
```