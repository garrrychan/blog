Title: March Madness
Date: 2019-04-18 12:00
Topic: EDA, Modeling and Evaluation
Slug: ncaa

How likely is a team to make the Final Four of the NCAA Tournament?

# Introduction

Each year, close to $4 billion is wagered on the NCAA Division 1 men's basketball tournament. Most of that money is wagered where the objective is to correctly predict winners of each game, with emphasis on the last four teams remaining (the Final Four). 

In this project, my motivation is the following:

Based on a college's regular season performance and seeding information, can I predict whether or not they will reach the final four? 

What are the variables that are correlated to predicting teams that make it to the final four? As a corollary, my model will also output the associated probabilities of making it to the final 4. Am I able to outperform a naive model? As even the sport pundits will tell you, since 2008, 53% of the time, at least 2 No. 1 seeds make the final four. So just by choosing two No 1. seeds, you're half way there.


As a trusted advisor to Coach Krzyzewski (Duke) or Coach Izzo (Michigan State), how would I recommend spending time developing the team? 

Or at the very least, how do I improve my 2020 bracket to make some money?

---

# Preprocessing

## Import libraries


```python
# data wrangling
import pandas as pd
import numpy as np

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
from IPython.display import HTML

# preprocessing & feature engineering
from sklearn.preprocessing import StandardScaler, LabelBinarizer, PolynomialFeatures
from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# modelling & evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix
# scientific notation off
np.set_printoptions(suppress=True)
# pd.options.display.float_format = '{:.2f}'.format

# suppress warnings
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
```

    /Users/gc/anaconda3/lib/python3.6/site-packages/sklearn/externals/six.py:31: DeprecationWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", DeprecationWarning)


---
## Data

The data spans 2003 to 2017 and is compiled from [sports-reference.com](https://www.sports-reference.com/cbb/schools/virginia/2019-gamelogs.html). 

The data is spread across four files:

- `regular_season.csv` - gamelogs for every regular season game
- `teams.csv` - team_id, names, and conferences
- `march_madness.csv` - gamelogs for each NCAA tournament game
- `march_madness_seeds.csv` - entry seeds for each team (W, X, Y, Z indicate the region)

### Understand the data


```python
regular = pd.read_csv("./data/ncaa_data/regular_season.csv")
HTML(regular.tail(3).to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>day_in_season</th>
      <th>winning_team_id</th>
      <th>winning_team_score</th>
      <th>losing_team_id</th>
      <th>losing_team_score</th>
      <th>winning_team_field_goals</th>
      <th>winning_team_field_goals_attempted</th>
      <th>winning_team_three_points</th>
      <th>winning_team_three_points_attempted</th>
      <th>winning_team_free_throws</th>
      <th>winning_team_free_throws_attempted</th>
      <th>winning_team_offensive_rebounds</th>
      <th>winning_team_defensive_rebounds</th>
      <th>winning_team_assists</th>
      <th>winning_team_turnovers</th>
      <th>winning_team_steals</th>
      <th>winning_team_blocks</th>
      <th>winning_team_personal_fouls</th>
      <th>losing_team_field_goals</th>
      <th>losing_team_field_goals_attempted</th>
      <th>losing_team_three_points</th>
      <th>losing_team_three_points_attempted</th>
      <th>losing_team_free_throws</th>
      <th>losing_team_free_throws_attempted</th>
      <th>losing_team_offensive_rebounds</th>
      <th>losing_team_defensive_rebounds</th>
      <th>losing_team_assists</th>
      <th>losing_team_turnovers</th>
      <th>losing_team_steals</th>
      <th>losing_team_blocks</th>
      <th>losing_team_personal_fouls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76633</th>
      <td>2017</td>
      <td>132</td>
      <td>1348</td>
      <td>70</td>
      <td>1433</td>
      <td>63</td>
      <td>24</td>
      <td>54</td>
      <td>8</td>
      <td>20</td>
      <td>14</td>
      <td>19</td>
      <td>9</td>
      <td>27</td>
      <td>12</td>
      <td>6</td>
      <td>3</td>
      <td>7</td>
      <td>18</td>
      <td>21</td>
      <td>67</td>
      <td>4</td>
      <td>14</td>
      <td>17</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
      <td>8</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>16</td>
    </tr>
    <tr>
      <th>76634</th>
      <td>2017</td>
      <td>132</td>
      <td>1374</td>
      <td>71</td>
      <td>1153</td>
      <td>56</td>
      <td>26</td>
      <td>52</td>
      <td>10</td>
      <td>19</td>
      <td>9</td>
      <td>13</td>
      <td>7</td>
      <td>27</td>
      <td>14</td>
      <td>8</td>
      <td>2</td>
      <td>6</td>
      <td>15</td>
      <td>19</td>
      <td>61</td>
      <td>4</td>
      <td>24</td>
      <td>14</td>
      <td>18</td>
      <td>17</td>
      <td>22</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>1</td>
      <td>13</td>
    </tr>
    <tr>
      <th>76635</th>
      <td>2017</td>
      <td>132</td>
      <td>1407</td>
      <td>59</td>
      <td>1402</td>
      <td>53</td>
      <td>21</td>
      <td>60</td>
      <td>1</td>
      <td>17</td>
      <td>16</td>
      <td>19</td>
      <td>14</td>
      <td>19</td>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>10</td>
      <td>20</td>
      <td>48</td>
      <td>6</td>
      <td>17</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>27</td>
      <td>10</td>
      <td>17</td>
      <td>1</td>
      <td>7</td>
      <td>18</td>
    </tr>
  </tbody>
</table>



Let's understand the last row. 
1407 = Troy, 1402 = Texas State.
Troy beat Texas State 59 to 53 on 132 day in season which is Sunday, March 2017. (Technically, 2016 - 2017 Season). Notice that all games finish on day 132 in each year.


```python
# regular.info()
# regular.describe()
```

* 32 variables
* games from 2003 to 2017
* stats per each team, team id
* no null values, all ints
* 2018 is left out in this data set


```python
teams = pd.read_csv("./data/ncaa_data/teams.csv")
HTML(teams.head(3).to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>conference_code</th>
      <th>conference_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>1101</td>
      <td>Abilene Chr</td>
      <td>southland</td>
      <td>Southland Conference</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>1101</td>
      <td>Abilene Chr</td>
      <td>southland</td>
      <td>Southland Conference</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>1101</td>
      <td>Abilene Chr</td>
      <td>southland</td>
      <td>Southland Conference</td>
    </tr>
  </tbody>
</table>




```python
teams = teams.drop(columns=["conference_name"])
```

* use to join team name, conference code to easily interpret
* drop conference name, it's redundant
* conference info from 2003 to 2018
* no null values, int and object looks good


```python
mm = pd.read_csv("./data/ncaa_data/march_madness.csv")
HTML(mm.head(3).to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>day_in_season</th>
      <th>winning_team_id</th>
      <th>winning_team_score</th>
      <th>losing_team_id</th>
      <th>losing_team_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>136</td>
      <td>1116</td>
      <td>63</td>
      <td>1234</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>136</td>
      <td>1120</td>
      <td>59</td>
      <td>1345</td>
      <td>58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>136</td>
      <td>1207</td>
      <td>68</td>
      <td>1250</td>
      <td>43</td>
    </tr>
  </tbody>
</table>



* 1985 to 2017 data about game logs for NCAA tournament
* scores, winning team & losing team 
* use 1985 -> 2017 data to predict 2018
* day 134, 135 are for first four, associated with 'a', 'b' seeds, they play to finalize seeding for final 64
* e.g. 11 Wake Forest 88, 11 Kansas State 95
* first day of round of 64 is 136
* last game is 154, which is the national champion
* e.g. UNC beat Gonzaga on Day 154 in 2017
* Let's drop 1985->2002 data, since I don't have regular season, seed or conference information for those years.


```python
mm = mm[mm["season"]>=2003]
seeds = pd.read_csv("./data/ncaa_data/march_madness_seeds.csv")
HTML(seeds.head(3).to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>seed</th>
      <th>team_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>W01</td>
      <td>1328</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>W02</td>
      <td>1448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>W03</td>
      <td>1393</td>
    </tr>
  </tbody>
</table>



* seed information from 2003 to 2018
* there are 68 seeds
* Starting in 2011, NCAA tournament starts with 68 teams (e.g. 68 seeds), then dwindles down to 64.
* 8 lowest seeded teams play in the ‘first four’, and then the winners, come out to be apart of the 64.
* merge to use seed as a variable, join on team_id
* seed needs to be an int; let's remove the region, it doesn't really matter, what we care about are the raw seeds, treat Y16a and Y16b as 16


```python
seeds["seed"] = seeds["seed"].apply(lambda x: int(x[1:3]))
```


```python
seed_and_names = pd.merge(seeds, teams, how="left", on=["season","team_id"]).drop_duplicates()
# 996 unique teams
HTML(seed_and_names.head(3).to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>seed</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>conference_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1</td>
      <td>1328</td>
      <td>Oklahoma</td>
      <td>big_twelve</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2003</td>
      <td>2</td>
      <td>1448</td>
      <td>Wake Forest</td>
      <td>acc</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2003</td>
      <td>3</td>
      <td>1393</td>
      <td>Syracuse</td>
      <td>big_east</td>
    </tr>
  </tbody>
</table>



* I need a label to indicate if the team is in the final 4
* e.g. in 2017, final four were Gonzaga, South Carolina, Oregon and North Carolina
* 1211, 1376, 1322, 1314 respectively
* So, we can see that on day 154, the national championship game was played, and the winner was North Carolina
* Moreover, the final four teams are North Carolina vs Oregon, and Gonzaga vs South Carolina
* These correspond to games played on 152


```python
HTML(mm[mm["season"]==2017].tail().to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>day_in_season</th>
      <th>winning_team_id</th>
      <th>winning_team_score</th>
      <th>losing_team_id</th>
      <th>losing_team_score</th>
      <th>final_four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2112</th>
      <td>2017</td>
      <td>146</td>
      <td>1314</td>
      <td>75</td>
      <td>1246</td>
      <td>73</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2113</th>
      <td>2017</td>
      <td>146</td>
      <td>1376</td>
      <td>77</td>
      <td>1196</td>
      <td>70</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2114</th>
      <td>2017</td>
      <td>152</td>
      <td>1211</td>
      <td>77</td>
      <td>1376</td>
      <td>73</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2115</th>
      <td>2017</td>
      <td>152</td>
      <td>1314</td>
      <td>77</td>
      <td>1332</td>
      <td>76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2116</th>
      <td>2017</td>
      <td>154</td>
      <td>1314</td>
      <td>71</td>
      <td>1211</td>
      <td>65</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



We can see the championship game was also played on 154, final four played on 152! 
Perfect, now we can flag all of these teams as final four = 1

We'll query the winning_team_id and losing_team_id for each of those games to get the final four teams


```python
# 15 seasons, we should get 2*15 = 30 final four games
mm["final_four"]=mm["day_in_season"].apply(lambda x: 1 if x == 152 else 0)
```

---
## Define X and y

Let's define X and y so we can more easily perform EDA, feature engineering and modeling.
First, we'll need a target vector y, with all the teams in each season and whether or not
they made it to the Final Four.

Let's work off the seeds_and_names data frame, as those are all of our 996 participating colleges for each season.


```python
seed_and_names["final_four"]=np.zeros(len(seed_and_names)) # zeros

final_four_list = (list(zip(mm.query("final_four==1").season,mm.query("final_four==1").winning_team_id))+
     list(zip(mm.query("final_four==1").season,mm.query("final_four==1").losing_team_id)))

# fill in teams with 1 if final_four team
for season, team_id in final_four_list:
    seed_and_names["final_four"]+=np.where((seed_and_names.season==season) & (seed_and_names.team_id==team_id),1,0)
```


```python
HTML(seed_and_names.query("final_four==1 & season==2017").to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>seed</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>conference_code</th>
      <th>conference_name</th>
      <th>final_four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23045</th>
      <td>2017</td>
      <td>7</td>
      <td>1376</td>
      <td>South Carolina</td>
      <td>sec</td>
      <td>Southeastern Conference</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23292</th>
      <td>2017</td>
      <td>1</td>
      <td>1211</td>
      <td>Gonzaga</td>
      <td>wcc</td>
      <td>West Coast Conference</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23604</th>
      <td>2017</td>
      <td>3</td>
      <td>1332</td>
      <td>Oregon</td>
      <td>pac_twelve</td>
      <td>Pacific-12 Conference</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>23905</th>
      <td>2017</td>
      <td>1</td>
      <td>1314</td>
      <td>North Carolina</td>
      <td>acc</td>
      <td>Atlantic Coast Conference</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>



Great, now we have a final_four which will be our target y. Now we have to aggregate the stats for each
team, over all the 76636 games played.

e.g. for South Carolina, we now need the following aggregated stats for, points for, points against, etc.
Aggregating opponents stats so we have a sense of offensive and defensive ability because
defense wins champions!


```python
# for each time a a specific team won, sum up all of their points
# for each time a specific team lose, sum of all of their points
# sum up points from won and lost games

# let's use team 1102 (Airforce)
regular.groupby(["season","winning_team_id"]).sum().query("winning_team_id==1102 & season==2003")["winning_team_score"][0]
```




    825




```python
regular.groupby(["season","losing_team_id"]).sum().query("losing_team_id==1102 & season==2003")["losing_team_score"][0]
```




    778




```python
# also find out the points allowed by Airforce when they win
regular.groupby(["season","winning_team_id"]).sum().query("winning_team_id==1102 & season==2003")["losing_team_score"][0]
```




    638




```python
# also find out the points allowed by Airforce when they lose
regular.groupby(["season","losing_team_id"]).sum().query("losing_team_id==1102 & season==2003")["winning_team_score"][0]
# points allowed total 1596
```




    958




```python
# Let's write a function to automate this for any team and season
# Note, ran into error with Wichita St. since they were undefeated, 
# need to add try except clause, to avoid indexing issues

def points_for(team_id, season):
    '''This function generates the team's total points scored in a season.
    points_for(1102,2003) >>> 1603'''
    # points from winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}').values[0][1]
    except:
        i = 0
    # points from losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}').values[0][3]
    except:
        j = 0
    return i+j

def points_against(team_id,season):
    '''This function generates the team's points against in a season.
    points_against(1102,2003) >> 1596'''
    # points allowed from winning games
    try:
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}').values[0][3]
    except:
        i = 0
    # points allowed from losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}').values[0][2]
    except:
        j = 0
    return i+j
```


```python
%time
regular

np.where((regular.season==2003) & (regular.winning_team_id==1102), regular.winning_team_score,0).sum()
```

    CPU times: user 5 µs, sys: 1 µs, total: 6 µs
    Wall time: 10.3 µs





    825




```python
np.where((regular.season==2003) & (regular.losing_team_id==1102), regular.winning_team_score,0).sum()
```




    958




```python
958+825
```




    1783




```python
%time
points_for(1102,2003)
```

    CPU times: user 4 µs, sys: 1 µs, total: 5 µs
    Wall time: 10 µs





    1603




```python
regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id==1102 & season==2003')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>day_in_season</th>
      <th>winning_team_id</th>
      <th>winning_team_score</th>
      <th>losing_team_score</th>
      <th>winning_team_field_goals</th>
      <th>winning_team_field_goals_attempted</th>
      <th>winning_team_three_points</th>
      <th>winning_team_three_points_attempted</th>
      <th>winning_team_free_throws</th>
      <th>winning_team_free_throws_attempted</th>
      <th>...</th>
      <th>losing_team_three_points_attempted</th>
      <th>losing_team_free_throws</th>
      <th>losing_team_free_throws_attempted</th>
      <th>losing_team_offensive_rebounds</th>
      <th>losing_team_defensive_rebounds</th>
      <th>losing_team_assists</th>
      <th>losing_team_turnovers</th>
      <th>losing_team_steals</th>
      <th>losing_team_blocks</th>
      <th>losing_team_personal_fouls</th>
    </tr>
    <tr>
      <th>season</th>
      <th>losing_team_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2003</th>
      <th>1102</th>
      <td>1335</td>
      <td>21492</td>
      <td>958</td>
      <td>778</td>
      <td>312</td>
      <td>628</td>
      <td>76</td>
      <td>178</td>
      <td>258</td>
      <td>365</td>
      <td>...</td>
      <td>324</td>
      <td>149</td>
      <td>230</td>
      <td>71</td>
      <td>239</td>
      <td>161</td>
      <td>187</td>
      <td>79</td>
      <td>16</td>
      <td>332</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 30 columns</p>
</div>




```python
%%time 

seed_and_names["points_for"] = (seed_and_names[["team_id","season"]].
                                apply(lambda seed_and_names: 
                                      points_for(seed_and_names["team_id"],
                                                    seed_and_names["season"]),axis=1))

seed_and_names["points_against"] = (seed_and_names[["team_id","season"]].
                                    apply(lambda seed_and_names: 
                                          points_against(seed_and_names["team_id"],
                                                         seed_and_names["season"]),axis=1))
```

    CPU times: user 2min 53s, sys: 18.9 s, total: 3min 11s
    Wall time: 3min 54s



```python
seed_and_names
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>seed</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>conference_code</th>
      <th>final_four</th>
      <th>points_for</th>
      <th>points_against</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1</td>
      <td>1328</td>
      <td>Oklahoma</td>
      <td>big_twelve</td>
      <td>0.0</td>
      <td>2135</td>
      <td>1805</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2003</td>
      <td>2</td>
      <td>1448</td>
      <td>Wake Forest</td>
      <td>acc</td>
      <td>0.0</td>
      <td>2274</td>
      <td>1961</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2003</td>
      <td>3</td>
      <td>1393</td>
      <td>Syracuse</td>
      <td>big_east</td>
      <td>1.0</td>
      <td>2323</td>
      <td>2027</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2003</td>
      <td>4</td>
      <td>1257</td>
      <td>Louisville</td>
      <td>cusa</td>
      <td>0.0</td>
      <td>2455</td>
      <td>2054</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2003</td>
      <td>5</td>
      <td>1280</td>
      <td>Mississippi St</td>
      <td>sec</td>
      <td>0.0</td>
      <td>2105</td>
      <td>1805</td>
    </tr>
    <tr>
      <th>129</th>
      <td>2003</td>
      <td>6</td>
      <td>1329</td>
      <td>Oklahoma St</td>
      <td>big_twelve</td>
      <td>0.0</td>
      <td>1993</td>
      <td>1804</td>
    </tr>
    <tr>
      <th>151</th>
      <td>2003</td>
      <td>7</td>
      <td>1386</td>
      <td>St Joseph's PA</td>
      <td>a_ten</td>
      <td>0.0</td>
      <td>2048</td>
      <td>1719</td>
    </tr>
    <tr>
      <th>185</th>
      <td>2003</td>
      <td>8</td>
      <td>1143</td>
      <td>California</td>
      <td>pac_ten</td>
      <td>0.0</td>
      <td>2160</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2003</td>
      <td>9</td>
      <td>1301</td>
      <td>NC State</td>
      <td>acc</td>
      <td>0.0</td>
      <td>2172</td>
      <td>2040</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2003</td>
      <td>10</td>
      <td>1120</td>
      <td>Auburn</td>
      <td>sec</td>
      <td>0.0</td>
      <td>2103</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>280</th>
      <td>2003</td>
      <td>11</td>
      <td>1335</td>
      <td>Penn</td>
      <td>ivy</td>
      <td>0.0</td>
      <td>1895</td>
      <td>1604</td>
    </tr>
    <tr>
      <th>314</th>
      <td>2003</td>
      <td>12</td>
      <td>1139</td>
      <td>Butler</td>
      <td>horizon</td>
      <td>0.0</td>
      <td>1989</td>
      <td>1767</td>
    </tr>
    <tr>
      <th>325</th>
      <td>2003</td>
      <td>13</td>
      <td>1122</td>
      <td>Austin Peay</td>
      <td>ovc</td>
      <td>0.0</td>
      <td>1882</td>
      <td>1828</td>
    </tr>
    <tr>
      <th>359</th>
      <td>2003</td>
      <td>14</td>
      <td>1264</td>
      <td>Manhattan</td>
      <td>maac</td>
      <td>0.0</td>
      <td>2209</td>
      <td>1965</td>
    </tr>
    <tr>
      <th>393</th>
      <td>2003</td>
      <td>15</td>
      <td>1190</td>
      <td>ETSU</td>
      <td>southern</td>
      <td>0.0</td>
      <td>2152</td>
      <td>2092</td>
    </tr>
    <tr>
      <th>418</th>
      <td>2003</td>
      <td>16</td>
      <td>1354</td>
      <td>S Carolina St</td>
      <td>meac</td>
      <td>0.0</td>
      <td>2184</td>
      <td>2138</td>
    </tr>
    <tr>
      <th>452</th>
      <td>2003</td>
      <td>1</td>
      <td>1400</td>
      <td>Texas</td>
      <td>big_twelve</td>
      <td>1.0</td>
      <td>2208</td>
      <td>1923</td>
    </tr>
    <tr>
      <th>474</th>
      <td>2003</td>
      <td>2</td>
      <td>1196</td>
      <td>Florida</td>
      <td>sec</td>
      <td>0.0</td>
      <td>2353</td>
      <td>2005</td>
    </tr>
    <tr>
      <th>508</th>
      <td>2003</td>
      <td>3</td>
      <td>1462</td>
      <td>Xavier</td>
      <td>a_ten</td>
      <td>0.0</td>
      <td>2347</td>
      <td>1979</td>
    </tr>
    <tr>
      <th>526</th>
      <td>2003</td>
      <td>4</td>
      <td>1390</td>
      <td>Stanford</td>
      <td>pac_ten</td>
      <td>0.0</td>
      <td>2240</td>
      <td>2076</td>
    </tr>
    <tr>
      <th>553</th>
      <td>2003</td>
      <td>5</td>
      <td>1163</td>
      <td>Connecticut</td>
      <td>big_east</td>
      <td>0.0</td>
      <td>2401</td>
      <td>2142</td>
    </tr>
    <tr>
      <th>582</th>
      <td>2003</td>
      <td>6</td>
      <td>1268</td>
      <td>Maryland</td>
      <td>acc</td>
      <td>0.0</td>
      <td>2262</td>
      <td>1872</td>
    </tr>
    <tr>
      <th>612</th>
      <td>2003</td>
      <td>7</td>
      <td>1277</td>
      <td>Michigan St</td>
      <td>big_ten</td>
      <td>0.0</td>
      <td>2084</td>
      <td>1890</td>
    </tr>
    <tr>
      <th>646</th>
      <td>2003</td>
      <td>8</td>
      <td>1261</td>
      <td>LSU</td>
      <td>sec</td>
      <td>0.0</td>
      <td>2270</td>
      <td>1950</td>
    </tr>
    <tr>
      <th>680</th>
      <td>2003</td>
      <td>9</td>
      <td>1345</td>
      <td>Purdue</td>
      <td>big_ten</td>
      <td>0.0</td>
      <td>2009</td>
      <td>1827</td>
    </tr>
    <tr>
      <th>714</th>
      <td>2003</td>
      <td>10</td>
      <td>1160</td>
      <td>Colorado</td>
      <td>big_twelve</td>
      <td>0.0</td>
      <td>2341</td>
      <td>2172</td>
    </tr>
    <tr>
      <th>729</th>
      <td>2003</td>
      <td>11</td>
      <td>1423</td>
      <td>UNC Wilmington</td>
      <td>caa</td>
      <td>0.0</td>
      <td>2141</td>
      <td>1788</td>
    </tr>
    <tr>
      <th>762</th>
      <td>2003</td>
      <td>12</td>
      <td>1140</td>
      <td>BYU</td>
      <td>mwc</td>
      <td>0.0</td>
      <td>2246</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>774</th>
      <td>2003</td>
      <td>13</td>
      <td>1360</td>
      <td>San Diego</td>
      <td>wcc</td>
      <td>0.0</td>
      <td>2091</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>808</th>
      <td>2003</td>
      <td>14</td>
      <td>1407</td>
      <td>Troy</td>
      <td>a_sun</td>
      <td>0.0</td>
      <td>2381</td>
      <td>2117</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>23645</th>
      <td>2017</td>
      <td>5</td>
      <td>1235</td>
      <td>Iowa St</td>
      <td>big_twelve</td>
      <td>0.0</td>
      <td>2669</td>
      <td>2376</td>
    </tr>
    <tr>
      <th>23667</th>
      <td>2017</td>
      <td>6</td>
      <td>1166</td>
      <td>Creighton</td>
      <td>big_east</td>
      <td>0.0</td>
      <td>2691</td>
      <td>2396</td>
    </tr>
    <tr>
      <th>23672</th>
      <td>2017</td>
      <td>7</td>
      <td>1276</td>
      <td>Michigan</td>
      <td>big_ten</td>
      <td>0.0</td>
      <td>2618</td>
      <td>2294</td>
    </tr>
    <tr>
      <th>23706</th>
      <td>2017</td>
      <td>8</td>
      <td>1274</td>
      <td>Miami FL</td>
      <td>acc</td>
      <td>0.0</td>
      <td>2220</td>
      <td>2038</td>
    </tr>
    <tr>
      <th>23720</th>
      <td>2017</td>
      <td>9</td>
      <td>1277</td>
      <td>Michigan St</td>
      <td>big_ten</td>
      <td>0.0</td>
      <td>2367</td>
      <td>2256</td>
    </tr>
    <tr>
      <th>23754</th>
      <td>2017</td>
      <td>10</td>
      <td>1329</td>
      <td>Oklahoma St</td>
      <td>big_twelve</td>
      <td>0.0</td>
      <td>2636</td>
      <td>2412</td>
    </tr>
    <tr>
      <th>23776</th>
      <td>2017</td>
      <td>11</td>
      <td>1348</td>
      <td>Rhode Island</td>
      <td>a_ten</td>
      <td>0.0</td>
      <td>2421</td>
      <td>2141</td>
    </tr>
    <tr>
      <th>23810</th>
      <td>2017</td>
      <td>12</td>
      <td>1305</td>
      <td>Nevada</td>
      <td>mwc</td>
      <td>0.0</td>
      <td>2719</td>
      <td>2412</td>
    </tr>
    <tr>
      <th>23816</th>
      <td>2017</td>
      <td>13</td>
      <td>1436</td>
      <td>Vermont</td>
      <td>aec</td>
      <td>0.0</td>
      <td>2422</td>
      <td>2074</td>
    </tr>
    <tr>
      <th>23838</th>
      <td>2017</td>
      <td>14</td>
      <td>1233</td>
      <td>Iona</td>
      <td>maac</td>
      <td>0.0</td>
      <td>2736</td>
      <td>2596</td>
    </tr>
    <tr>
      <th>23872</th>
      <td>2017</td>
      <td>15</td>
      <td>1240</td>
      <td>Jacksonville St</td>
      <td>ovc</td>
      <td>0.0</td>
      <td>2209</td>
      <td>2181</td>
    </tr>
    <tr>
      <th>23887</th>
      <td>2017</td>
      <td>16</td>
      <td>1300</td>
      <td>NC Central</td>
      <td>meac</td>
      <td>0.0</td>
      <td>2200</td>
      <td>1917</td>
    </tr>
    <tr>
      <th>23894</th>
      <td>2017</td>
      <td>16</td>
      <td>1413</td>
      <td>UC Davis</td>
      <td>big_west</td>
      <td>0.0</td>
      <td>2220</td>
      <td>2204</td>
    </tr>
    <tr>
      <th>23905</th>
      <td>2017</td>
      <td>1</td>
      <td>1314</td>
      <td>North Carolina</td>
      <td>acc</td>
      <td>1.0</td>
      <td>2783</td>
      <td>2340</td>
    </tr>
    <tr>
      <th>23939</th>
      <td>2017</td>
      <td>2</td>
      <td>1246</td>
      <td>Kentucky</td>
      <td>sec</td>
      <td>0.0</td>
      <td>2922</td>
      <td>2434</td>
    </tr>
    <tr>
      <th>23973</th>
      <td>2017</td>
      <td>3</td>
      <td>1417</td>
      <td>UCLA</td>
      <td>pac_twelve</td>
      <td>0.0</td>
      <td>2982</td>
      <td>2486</td>
    </tr>
    <tr>
      <th>23980</th>
      <td>2017</td>
      <td>4</td>
      <td>1139</td>
      <td>Butler</td>
      <td>big_east</td>
      <td>0.0</td>
      <td>2365</td>
      <td>2120</td>
    </tr>
    <tr>
      <th>23985</th>
      <td>2017</td>
      <td>5</td>
      <td>1278</td>
      <td>Minnesota</td>
      <td>big_ten</td>
      <td>0.0</td>
      <td>2484</td>
      <td>2280</td>
    </tr>
    <tr>
      <th>24019</th>
      <td>2017</td>
      <td>6</td>
      <td>1153</td>
      <td>Cincinnati</td>
      <td>aac</td>
      <td>0.0</td>
      <td>2532</td>
      <td>2068</td>
    </tr>
    <tr>
      <th>24024</th>
      <td>2017</td>
      <td>7</td>
      <td>1173</td>
      <td>Dayton</td>
      <td>a_ten</td>
      <td>0.0</td>
      <td>2279</td>
      <td>2003</td>
    </tr>
    <tr>
      <th>24047</th>
      <td>2017</td>
      <td>8</td>
      <td>1116</td>
      <td>Arkansas</td>
      <td>sec</td>
      <td>0.0</td>
      <td>2713</td>
      <td>2516</td>
    </tr>
    <tr>
      <th>24074</th>
      <td>2017</td>
      <td>9</td>
      <td>1371</td>
      <td>Seton Hall</td>
      <td>big_east</td>
      <td>0.0</td>
      <td>2346</td>
      <td>2246</td>
    </tr>
    <tr>
      <th>24108</th>
      <td>2017</td>
      <td>10</td>
      <td>1455</td>
      <td>Wichita St</td>
      <td>mvc</td>
      <td>0.0</td>
      <td>2703</td>
      <td>2066</td>
    </tr>
    <tr>
      <th>24141</th>
      <td>2017</td>
      <td>11</td>
      <td>1243</td>
      <td>Kansas St</td>
      <td>big_twelve</td>
      <td>0.0</td>
      <td>2367</td>
      <td>2209</td>
    </tr>
    <tr>
      <th>24163</th>
      <td>2017</td>
      <td>11</td>
      <td>1448</td>
      <td>Wake Forest</td>
      <td>acc</td>
      <td>0.0</td>
      <td>2645</td>
      <td>2493</td>
    </tr>
    <tr>
      <th>24197</th>
      <td>2017</td>
      <td>12</td>
      <td>1292</td>
      <td>MTSU</td>
      <td>cusa</td>
      <td>0.0</td>
      <td>2449</td>
      <td>2089</td>
    </tr>
    <tr>
      <th>24202</th>
      <td>2017</td>
      <td>13</td>
      <td>1457</td>
      <td>Winthrop</td>
      <td>big_south</td>
      <td>0.0</td>
      <td>2379</td>
      <td>2136</td>
    </tr>
    <tr>
      <th>24234</th>
      <td>2017</td>
      <td>14</td>
      <td>1245</td>
      <td>Kent</td>
      <td>mac</td>
      <td>0.0</td>
      <td>2581</td>
      <td>2476</td>
    </tr>
    <tr>
      <th>24268</th>
      <td>2017</td>
      <td>15</td>
      <td>1297</td>
      <td>N Kentucky</td>
      <td>horizon</td>
      <td>0.0</td>
      <td>2396</td>
      <td>2299</td>
    </tr>
    <tr>
      <th>24271</th>
      <td>2017</td>
      <td>16</td>
      <td>1411</td>
      <td>TX Southern</td>
      <td>swac</td>
      <td>0.0</td>
      <td>2529</td>
      <td>2441</td>
    </tr>
  </tbody>
</table>
<p>996 rows × 8 columns</p>
</div>




```python

```


```python
Now that we have a formula, let's repeat this for all the other columns.
```


```python
def fg_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_field_goals"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_field_goals"][0]
    except:
        j = 0
    return i+j

def fg_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_field_goals"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_field_goals"][0]
    except:
        j = 0
    return i+j


def fga_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_field_goals_attempted"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_field_goals_attempted"][0]
    except:
        j = 0
    return i+j

def fga_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_field_goals_attempted"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_field_goals_attempted"][0]
    except:
        j = 0
    return i+j



seed_and_names["fg_for"] = (seed_and_names[["team_id","season"]].
                                apply(lambda seed_and_names: 
                                      fg_for(seed_and_names["team_id"],
                                                    seed_and_names["season"]),axis=1))

seed_and_names["fg_against"] = (seed_and_names[["team_id","season"]].
                                    apply(lambda seed_and_names: 
                                          fg_against(seed_and_names["team_id"],
                                                         seed_and_names["season"]),axis=1))


seed_and_names["fga_for"] = (seed_and_names[["team_id","season"]].
                                apply(lambda seed_and_names: 
                                      fga_for(seed_and_names["team_id"],
                                                    seed_and_names["season"]),axis=1))

seed_and_names["fga_against"] = (seed_and_names[["team_id","season"]].
                                    apply(lambda seed_and_names: 
                                          fga_against(seed_and_names["team_id"],
                                                         seed_and_names["season"]),axis=1))
```


```python
def threes_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_three_points"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_three_points"][0]
    except:
        j = 0
    return i+j

def threes_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_three_points"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_three_points"][0]
    except:
        j = 0
    return i+j

def threes_a_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_three_points_attempted"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_three_points_attempted"][0]
    except:
        j = 0
    return i+j

def threes_a_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_three_points_attempted"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_three_points__attempted"][0]
    except:
        j = 0
    return i+j


seed_and_names["3pm_for"] = (seed_and_names[["team_id","season"]].
                                apply(lambda seed_and_names: 
                                      threes_for(seed_and_names["team_id"],
                                                    seed_and_names["season"]),axis=1))

seed_and_names["3pm_against"] = (seed_and_names[["team_id","season"]].
                                    apply(lambda seed_and_names: 
                                          threes_against(seed_and_names["team_id"],
                                                         seed_and_names["season"]),axis=1))


seed_and_names["3pa_for"] = (seed_and_names[["team_id","season"]].
                                apply(lambda seed_and_names: 
                                      threes_a_for(seed_and_names["team_id"],
                                                    seed_and_names["season"]),axis=1))

seed_and_names["3pa_against"] = (seed_and_names[["team_id","season"]].
                                    apply(lambda seed_and_names: 
                                          threes_a_against(seed_and_names["team_id"],
                                                         seed_and_names["season"]),axis=1))
```


```python
def ft_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_free_throws"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_free_throws"][0]
    except:
        j = 0
    return i+j

def ft_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_free_throws"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_free_throws"][0]
    except:
        j = 0
    return i+j

def fta_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_free_throws_attempted"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_free_throws_attempted"][0]
    except:
        j = 0
    return i+j

def fta_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_free_throws_attempted"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_free_throws_attempted"][0]
    except:
        j = 0
    return i+j



seed_and_names["ft_for"] = (seed_and_names[["team_id","season"]].
                                apply(lambda seed_and_names: 
                                      ft_for(seed_and_names["team_id"],
                                                    seed_and_names["season"]),axis=1))

seed_and_names["ft_against"] = (seed_and_names[["team_id","season"]].
                                    apply(lambda seed_and_names: 
                                          ft_against(seed_and_names["team_id"],
                                                         seed_and_names["season"]),axis=1))



seed_and_names["fta_for"] = (seed_and_names[["team_id","season"]].
                                apply(lambda seed_and_names: 
                                      fta_for(seed_and_names["team_id"],
                                                    seed_and_names["season"]),axis=1))

seed_and_names["fta_against"] = (seed_and_names[["team_id","season"]].
                                    apply(lambda seed_and_names: 
                                          fta_against(seed_and_names["team_id"],
                                                         seed_and_names["season"]),axis=1))
```


```python
def offensive_rebounds_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_offensive_rebounds"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_offensive_rebounds"][0]
    except:
        j = 0
    return i+j

def offensive_rebounds_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_offensive_rebounds"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_offensive_rebounds"][0]
    except:
        j = 0
    return i+j

def defensive_rebounds_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_defensive_rebounds"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_defensive_rebounds"][0]
    except:
        j = 0
    return i+j

def defensive_rebounds_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_defensive_rebounds"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_defensive_rebounds"][0]
    except:
        j = 0
    return i+j


seed_and_names["off_rebounds_for"] = (seed_and_names[["team_id","season"]].
                                      apply(lambda seed_and_names: 
                                      offensive_rebounds_for(seed_and_names["team_id"],
                                                             seed_and_names["season"]),axis=1))

seed_and_names["off_rebounds_against"] = (seed_and_names[["team_id","season"]].
                                          apply(lambda seed_and_names: 
                                          offensive_rebounds_against(seed_and_names["team_id"],
                                                                     seed_and_names["season"]),axis=1))


seed_and_names["def_rebounds_for"] = (seed_and_names[["team_id","season"]].
                                      apply(lambda seed_and_names: 
                                            defensive_rebounds_for(seed_and_names["team_id"],
                                                                   seed_and_names["season"]),axis=1))

seed_and_names["def_rebounds_against"] = (seed_and_names[["team_id","season"]].
                                          apply(lambda seed_and_names: 
                                                defensive_rebounds_against(seed_and_names["team_id"],
                                                                           seed_and_names["season"]),axis=1))
```


```python
def assists_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_assists"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_assists"][0]
    except:
        j = 0
    return i+j

def assists_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_assists"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_assists"][0]
    except:
        j = 0
    return i+j


seed_and_names["assists_for"] = (seed_and_names[["team_id","season"]].
                                      apply(lambda seed_and_names: 
                                      assists_for(seed_and_names["team_id"],
                                                             seed_and_names["season"]),axis=1))

seed_and_names["assists_against"] = (seed_and_names[["team_id","season"]].
                                          apply(lambda seed_and_names: 
                                          assists_against(seed_and_names["team_id"],
                                                                     seed_and_names["season"]),axis=1))
```


```python
def steals_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_steals"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_steals"][0]
    except:
        j = 0
    return i+j

def steals_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_steals"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_steals"][0]
    except:
        j = 0
    return i+j

def blocks_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_blocks"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_blocks"][0]
    except:
        j = 0
    return i+j

def blocks_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_blocks"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_blocks"][0]
    except:
        j = 0
    return i+j


seed_and_names["steals_for"] = (seed_and_names[["team_id","season"]].
                                      apply(lambda seed_and_names: 
                                      steals_for(seed_and_names["team_id"],
                                                             seed_and_names["season"]),axis=1))

seed_and_names["steals_against"] = (seed_and_names[["team_id","season"]].
                                          apply(lambda seed_and_names: 
                                          steals_against(seed_and_names["team_id"],
                                                                     seed_and_names["season"]),axis=1))


seed_and_names["blocks_for"] = (seed_and_names[["team_id","season"]].
                                      apply(lambda seed_and_names: 
                                            blocks_for(seed_and_names["team_id"],
                                                                   seed_and_names["season"]),axis=1))

seed_and_names["blocks_against"] = (seed_and_names[["team_id","season"]].
                                          apply(lambda seed_and_names: 
                                                blocks_against(seed_and_names["team_id"],
                                                               seed_and_names["season"]),axis=1))
```


```python
def turnovers_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_turnovers"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_turnovers"][0]
    except:
        j = 0
    return i+j

def turnovers_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_turnovers"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_turnovers"][0]
    except:
        j = 0
    return i+j

def fouls_for(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["winning_team_personal_fouls"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["losing_team_personal_fouls"][0]
    except:
        j = 0
    return i+j

def fouls_against(team_id, season):
    #winning games
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["losing_team_personal_fouls"][0]
    except:
        i = 0
    #losing games
    try:
        j = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["winning_team_personal_fouls"][0]
    except:
        j = 0
    return i+j


seed_and_names["turnovers_for"] = (seed_and_names[["team_id","season"]].
                                      apply(lambda seed_and_names: 
                                      turnovers_for(seed_and_names["team_id"],
                                                             seed_and_names["season"]),axis=1))

seed_and_names["turnovers_against"] = (seed_and_names[["team_id","season"]].
                                          apply(lambda seed_and_names: 
                                          turnovers_against(seed_and_names["team_id"],
                                                                     seed_and_names["season"]),axis=1))


seed_and_names["fouls_for"] = (seed_and_names[["team_id","season"]].
                                      apply(lambda seed_and_names: 
                                            fouls_for(seed_and_names["team_id"],
                                                                   seed_and_names["season"]),axis=1))

seed_and_names["fouls_against"] = (seed_and_names[["team_id","season"]].
                                          apply(lambda seed_and_names: 
                                                fouls_against(seed_and_names["team_id"],
                                                               seed_and_names["season"]),axis=1))
```


```python
regular_season_total = seed_and_names.reset_index().drop(columns=["index"])
```


```python
regular_season_total.columns
```




    Index(['season', 'seed', 'team_id', 'team_name', 'conference_code',
           'final_four', 'points_for', 'points_against', 'fg_for', 'fg_against',
           'fga_for', 'fga_against', '3pm_for', '3pm_against', '3pa_for',
           '3pa_against', 'ft_for', 'ft_against', 'fta_for', 'fta_against',
           'off_rebounds_for', 'off_rebounds_against', 'def_rebounds_for',
           'def_rebounds_against', 'assists_for', 'assists_against', 'steals_for',
           'steals_against', 'blocks_for', 'blocks_against', 'turnovers_for',
           'turnovers_against', 'fouls_for', 'fouls_against'],
          dtype='object')



---
## Train Test Split


* As always, let's train test split (80% / 20%). Given it's time series we can split on years.
* There are 15 years of data, so we'll train on 12 years and test on the final 3 years (2015, 2016, 2017)


```python
y_test = regular_season_total[regular_season_total["season"]>=2015]["final_four"]
y_train = regular_season_total[regular_season_total["season"]<2015]["final_four"]
```


```python
X = regular_season_total[['season', 'seed', 'team_id', 'team_name', 'conference_code',
                           'points_for', 'points_against', 'fg_for', 'fg_against',
                           '3pm_for', '3pm_against', 'fga_for', 'fga_against', '3pa_for',
                           '3pa_against', 'ft_for', 'ft_against', 'fta_for', 'fta_against',
                           'off_rebounds_for', 'off_rebounds_against', 'def_rebounds_for',
                           'def_rebounds_against', 'assists_for', 'assists_against', 'steals_for',
                           'steals_against', 'blocks_for', 'blocks_against', 'turnovers_for',
                           'turnovers_against', 'fouls_for', 'fouls_against']]
```


```python
X_train = X.query("season<2015")
X_test = X.query("season>=2015")
```

---
## Imputation
* Data is pretty complete and in the right type which is great. We'll just need to create multi class labels, such that we can identify which conferences a team is in, as I suspect that some teams play in more competitive conferences than others.
* Let's label qualitative variables.
* We'll standardize after splitting.


```python
mapper = DataFrameMapper([
    (['season'],None),
    (['seed'],None),
    (['team_id'],None),
    (['conference_code'],LabelBinarizer()),    
    (['points_for'],None),
    (['points_against'],None),
    (['fg_for'],None),
    (['fg_against'],None),
    (['3pm_for'],None), 
    (['3pm_against'],None), 
    (['fga_for'],None),
    (['fga_against'],None),
    (['3pa_for'],None), 
    (['3pa_against'],None),
    (['ft_for'],None),  
    (['ft_against'],None),
    (['fta_for'],None),
    (['fta_against'],None),
    (['off_rebounds_for'],None),
    (['off_rebounds_against'],None),
    (['def_rebounds_for'],None),
    (['def_rebounds_against'],None),
    (['assists_for'],None),
    (['assists_against'],None),
    (['steals_for'],None),
    (['steals_against'],None),
    (['blocks_for'],None),
    (['blocks_against'],None),
    (['turnovers_for'],None),
    (['turnovers_against'],None),
    (['fouls_for'],None),
    (['fouls_against'],None)
],df_out=True)
```


```python
Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test) # you never want to fit because if you see something you never saw before emoji
# then it will be 'labelbinarized', when it fact it should be treated as something diff instead
```


```python
# problem
# list(zip(Z_train.columns,Z_test.columns)) # there are less conferences in 2015 -> 2018
# mid_cont, pac_ten are in Z_train but not in Z_test
```


```python
teams[teams["conference_code"]=="mid_cont"].drop_duplicates(subset="season") # only occured 2004->2007
# Chicago moved from mid_cont to gwc in 2010
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>conference_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14628</th>
      <td>2004</td>
      <td>1147</td>
      <td>Centenary</td>
      <td>mid_cont</td>
    </tr>
    <tr>
      <th>14629</th>
      <td>2005</td>
      <td>1147</td>
      <td>Centenary</td>
      <td>mid_cont</td>
    </tr>
    <tr>
      <th>14630</th>
      <td>2006</td>
      <td>1147</td>
      <td>Centenary</td>
      <td>mid_cont</td>
    </tr>
    <tr>
      <th>14631</th>
      <td>2007</td>
      <td>1147</td>
      <td>Centenary</td>
      <td>mid_cont</td>
    </tr>
    <tr>
      <th>16279</th>
      <td>2003</td>
      <td>1152</td>
      <td>Chicago St</td>
      <td>mid_cont</td>
    </tr>
  </tbody>
</table>
</div>




```python
teams[teams["conference_code"]=="pac_ten"].drop_duplicates(subset="season") # only occured 2003->2011;
# in 2012, Arizona played in pac_twelmanually add ve
# let's be more verbose and label mid_cont in pac_ten columns in test, with 0 so they're the same width
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>team_id</th>
      <th>team_name</th>
      <th>conference_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3601</th>
      <td>2003</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3602</th>
      <td>2004</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3603</th>
      <td>2005</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3604</th>
      <td>2006</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3605</th>
      <td>2007</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3606</th>
      <td>2008</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3607</th>
      <td>2009</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3608</th>
      <td>2010</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
    <tr>
      <th>3609</th>
      <td>2011</td>
      <td>1112</td>
      <td>Arizona</td>
      <td>pac_ten</td>
    </tr>
  </tbody>
</table>
</div>




```python
Z_test["conference_code_pac_ten"] = pd.DataFrame(np.zeros((Z_test.shape[0],1)))
Z_test["conference_code_mid_cont"] = pd.DataFrame(np.zeros((Z_test.shape[0],1)))

#Showing up as NaN -> need to make zero
Z_test["conference_code_pac_ten"].fillna(0,inplace=True)
Z_test["conference_code_mid_cont"].fillna(0,inplace=True)
```

---
## Establish Benchmarks
Let's quickly spin up a naive baseline/benchmark model & naive out the door. Then, in subsequent steps, we can feature engineering and model to improve our score!

From FiveThirtyEight, 1 seeds only had a 35%-52% of reaching the final four.

<img src="./images/ncaa.png"  width="620" height="120">


Let's see how how our naive model's prediction fared.


```python
# Baseline logistic 

# Step 1: Instantiate our model.
logreg_baseline = LogisticRegression(random_state=8)

# Step 2: Fit our model.
logreg_baseline.fit(Z_train,y_train)

# Step 3 (part 1): Generate prediction values
print(f'Number of teams making it to final four: {sum(logreg_baseline.predict(Z_train))}')

# Step 3 (part 2): Generate predictions/probabilities
logreg_baseline.predict_proba(Z_train)[:,1]

# Step 4: Score the model:
print(f' Logreg train accuracy: {logreg_baseline.score(Z_train,y_train)}')
print(f' Logreg test accuracy: {logreg_baseline.score(Z_test,y_test)}')
```

    Number of teams making it to final four: 20
     Logreg train accuracy: 0.946969696969697
     Logreg test accuracy: 0.9656862745098039


---
### Imbalanced Classes

* Accuracy of 94% looks good right? However, we have imbalanced classes.
* Over 15 years, 60 final four teams, 936 non-final four, 996 total.
* 1-60/996 = 94%
* Any naive model would be equally as good as this baseline logistic regression, by simply taking the accuracy.
* This is a misleading accuracy as it's overestimated. Let's quickly spin up a AUC score to score the accuracy of this logistic classifier, and measure this imbalance.


```python
roc_auc_score(y_test, logreg_baseline.predict_proba(Z_test)[:,1])
```




    0.7777777777777778



See how the area under the curve score is close to 50%. With this model (trained on imbalanced classes), we are no better than a 'no information' classifer.

To get a better metric of our baseline model, let's better represent the minority class (final four)
with more signal, via SMOTE (Synthetic Minority Over-Sampling Technique).


```python
# added final_four to this mapper so I can split after, preserve order rather than randomly splitting
mapper_all = DataFrameMapper([
    (['season'],None),
    (['seed'],None),
    (['team_id'],None),
    (['conference_code'],LabelBinarizer()),    
    (['final_four'],None),
    (['points_for'],None),
    (['points_against'],None),
    (['fg_for'],None),
    (['fg_against'],None),
    (['3pm_for'],None), 
    (['3pm_against'],None), 
    (['fga_for'],None),
    (['fga_against'],None),
    (['3pa_for'],None), 
    (['3pa_against'],None),
    (['ft_for'],None),  
    (['ft_against'],None),
    (['fta_for'],None),
    (['fta_against'],None),
    (['off_rebounds_for'],None),
    (['off_rebounds_against'],None),
    (['def_rebounds_for'],None),
    (['def_rebounds_against'],None),
    (['assists_for'],None),
    (['assists_against'],None),
    (['steals_for'],None),
    (['steals_against'],None),
    (['blocks_for'],None),
    (['blocks_against'],None),
    (['turnovers_for'],None),
    (['turnovers_against'],None),
    (['fouls_for'],None),
    (['fouls_against'],None)
],df_out=True)
```


```python
Z = mapper_all.fit_transform(regular_season_total)
y = Z["final_four"]
X = Z[['season', 'seed', 'team_id', 'conference_code_a_sun',
       'conference_code_a_ten', 'conference_code_aac', 'conference_code_acc',
       'conference_code_aec', 'conference_code_big_east',
       'conference_code_big_sky', 'conference_code_big_south',
       'conference_code_big_ten', 'conference_code_big_twelve',
       'conference_code_big_west', 'conference_code_caa',
       'conference_code_cusa', 'conference_code_horizon',
       'conference_code_ivy', 'conference_code_maac', 'conference_code_mac',
       'conference_code_meac', 'conference_code_mid_cont',
       'conference_code_mvc', 'conference_code_mwc', 'conference_code_nec',
       'conference_code_ovc', 'conference_code_pac_ten',
       'conference_code_pac_twelve', 'conference_code_patriot',
       'conference_code_sec', 'conference_code_southern',
       'conference_code_southland', 'conference_code_summit',
       'conference_code_sun_belt', 'conference_code_swac',
       'conference_code_wac', 'conference_code_wcc',
       'points_for', 'points_against', 'fg_for', 'fg_against', '3pm_for',
       '3pm_against', 'fga_for', 'fga_against', '3pa_for', '3pa_against',
       'ft_for', 'ft_against', 'fta_for', 'fta_against', 'off_rebounds_for',
       'off_rebounds_against', 'def_rebounds_for', 'def_rebounds_against',
       'assists_for', 'assists_against', 'steals_for', 'steals_against',
       'blocks_for', 'blocks_against', 'turnovers_for', 'turnovers_against',
       'fouls_for', 'fouls_against']]

# instantiate SMOTE
smote = SMOTE(ratio='minority',random_state=8)
X_sm, y_sm = smote.fit_sample(X, y)
print(X_sm.shape) 
print(y_sm.shape) # there are 936 final four rows and 936 non-final four
print(sum(y_sm))
```

    (1872, 65)
    (1872,)
    936


Let's recalculate accuracy and auc


```python
df = pd.concat([pd.DataFrame(y_sm),pd.DataFrame(X_sm)],axis=1)
df.columns = ['final_four','season', 'seed', 'team_id', 'conference_code_a_sun',
       'conference_code_a_ten', 'conference_code_aac', 'conference_code_acc',
       'conference_code_aec', 'conference_code_big_east',
       'conference_code_big_sky', 'conference_code_big_south',
       'conference_code_big_ten', 'conference_code_big_twelve',
       'conference_code_big_west', 'conference_code_caa',
       'conference_code_cusa', 'conference_code_horizon',
       'conference_code_ivy', 'conference_code_maac', 'conference_code_mac',
       'conference_code_meac', 'conference_code_mid_cont',
       'conference_code_mvc', 'conference_code_mwc', 'conference_code_nec',
       'conference_code_ovc', 'conference_code_pac_ten',
       'conference_code_pac_twelve', 'conference_code_patriot',
       'conference_code_sec', 'conference_code_southern',
       'conference_code_southland', 'conference_code_summit',
       'conference_code_sun_belt', 'conference_code_swac',
       'conference_code_wac', 'conference_code_wcc', 'points_for',
       'points_against', 'fg_for', 'fg_against', '3pm_for', '3pm_against',
       'fga_for', 'fga_against', '3pa_for', '3pa_against', 'ft_for',
       'ft_against', 'fta_for', 'fta_against', 'off_rebounds_for',
       'off_rebounds_against', 'def_rebounds_for', 'def_rebounds_against',
       'assists_for', 'assists_against', 'steals_for', 'steals_against',
       'blocks_for', 'blocks_against', 'turnovers_for', 'turnovers_against',
       'fouls_for', 'fouls_against']

# Let's split on the years again
y_test = df[df["season"]>=2015]["final_four"]
y_train = df[df["season"]<2015]["final_four"]

# 269/1603 so about 17%/83% split which is fine, I prefer the interpretability and preserve meaning.
# We are using 2003 -> 2014 data to predict and test 2015, 2016, 2017.
```


```python
X = df.loc[:,"season":"fouls_against"]
X_test = X[X["season"]>=2015]
X_train = X[X["season"]<2015]
```


```python
print(X_train.shape)
print(y_train.shape)
print("\n")
print(X_test.shape)
print(y_test.shape)
```

    (1603, 65)
    (1603,)
    
    
    (269, 65)
    (269,)



```python
# Baseline logistic 

# Step 1: Instantiate our model.
logreg_baseline = LogisticRegression(random_state=8)

# Step 2: Fit our model.
logreg_baseline.fit(X_train,y_train)

# Step 3 (part 1): Generate prediction values
print(f'Number of teams making it to final four: {sum(logreg_baseline.predict(X_train))}')

# Step 3 (part 2): Generate predictions/probabilities
logreg_baseline.predict_proba(X_train)[:,1]

# Step 4: Score the model:
print(f' Logreg train accuracy: {cross_val_score(logreg_baseline, X_train, y_train, cv=5).mean()}')
print(f' Logreg test accuracy: {cross_val_score(logreg_baseline, X_test, y_test, cv=5).mean()}')
```

    Number of teams making it to final four: 837
     Logreg train accuracy: 0.9532202462914677
     Logreg test accuracy: 0.9221269296740996



```python
roc_auc_score(y_test, logreg_baseline.predict_proba(X_test)[:,1])
```




    0.9602949134199135




```python
print(f' seed coeff {np.exp(-0.608454)}')
print(f' acc coeff {np.exp(-3.538678)}')
```

     seed coeff 0.5441915391886456
     acc coeff 0.029051708064839428



```python
pd.DataFrame(list(zip(X.columns,logreg_baseline.coef_.T[:,0])))
# The seed is very important.
# Remember, need to take exponential. If your seed increases by 1, your likelihood of reaching final four
# is 0.54x less
# The individual team stats are less important, as I suspect the seed already captures a lot of that information.

# And what conference you play in matters (acc, big_twelve or big_ten)
# If you play in acc, your likelihood of reaching final four is 0.03x less, this is probably explaining
# for the teams who are not top tier.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>season</td>
      <td>-0.001062</td>
    </tr>
    <tr>
      <th>1</th>
      <td>seed</td>
      <td>-0.439209</td>
    </tr>
    <tr>
      <th>2</th>
      <td>team_id</td>
      <td>-0.000361</td>
    </tr>
    <tr>
      <th>3</th>
      <td>conference_code_a_sun</td>
      <td>-0.039266</td>
    </tr>
    <tr>
      <th>4</th>
      <td>conference_code_a_ten</td>
      <td>-1.133449</td>
    </tr>
    <tr>
      <th>5</th>
      <td>conference_code_aac</td>
      <td>-0.078609</td>
    </tr>
    <tr>
      <th>6</th>
      <td>conference_code_acc</td>
      <td>-1.668645</td>
    </tr>
    <tr>
      <th>7</th>
      <td>conference_code_aec</td>
      <td>-0.026041</td>
    </tr>
    <tr>
      <th>8</th>
      <td>conference_code_big_east</td>
      <td>-1.204675</td>
    </tr>
    <tr>
      <th>9</th>
      <td>conference_code_big_sky</td>
      <td>-0.012483</td>
    </tr>
    <tr>
      <th>10</th>
      <td>conference_code_big_south</td>
      <td>-0.025224</td>
    </tr>
    <tr>
      <th>11</th>
      <td>conference_code_big_ten</td>
      <td>-0.618794</td>
    </tr>
    <tr>
      <th>12</th>
      <td>conference_code_big_twelve</td>
      <td>-1.790391</td>
    </tr>
    <tr>
      <th>13</th>
      <td>conference_code_big_west</td>
      <td>-0.035615</td>
    </tr>
    <tr>
      <th>14</th>
      <td>conference_code_caa</td>
      <td>0.459960</td>
    </tr>
    <tr>
      <th>15</th>
      <td>conference_code_cusa</td>
      <td>-0.361446</td>
    </tr>
    <tr>
      <th>16</th>
      <td>conference_code_horizon</td>
      <td>0.509026</td>
    </tr>
    <tr>
      <th>17</th>
      <td>conference_code_ivy</td>
      <td>-0.021336</td>
    </tr>
    <tr>
      <th>18</th>
      <td>conference_code_maac</td>
      <td>-0.036216</td>
    </tr>
    <tr>
      <th>19</th>
      <td>conference_code_mac</td>
      <td>-0.045925</td>
    </tr>
    <tr>
      <th>20</th>
      <td>conference_code_meac</td>
      <td>-0.014014</td>
    </tr>
    <tr>
      <th>21</th>
      <td>conference_code_mid_cont</td>
      <td>-0.008154</td>
    </tr>
    <tr>
      <th>22</th>
      <td>conference_code_mvc</td>
      <td>-0.398462</td>
    </tr>
    <tr>
      <th>23</th>
      <td>conference_code_mwc</td>
      <td>-0.824229</td>
    </tr>
    <tr>
      <th>24</th>
      <td>conference_code_nec</td>
      <td>-0.006708</td>
    </tr>
    <tr>
      <th>25</th>
      <td>conference_code_ovc</td>
      <td>-0.091618</td>
    </tr>
    <tr>
      <th>26</th>
      <td>conference_code_pac_ten</td>
      <td>-0.637526</td>
    </tr>
    <tr>
      <th>27</th>
      <td>conference_code_pac_twelve</td>
      <td>-0.575800</td>
    </tr>
    <tr>
      <th>28</th>
      <td>conference_code_patriot</td>
      <td>-0.022216</td>
    </tr>
    <tr>
      <th>29</th>
      <td>conference_code_sec</td>
      <td>-0.921589</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>conference_code_wac</td>
      <td>-0.123001</td>
    </tr>
    <tr>
      <th>36</th>
      <td>conference_code_wcc</td>
      <td>-0.881598</td>
    </tr>
    <tr>
      <th>37</th>
      <td>points_for</td>
      <td>1.961381</td>
    </tr>
    <tr>
      <th>38</th>
      <td>points_against</td>
      <td>1.921633</td>
    </tr>
    <tr>
      <th>39</th>
      <td>fg_for</td>
      <td>-3.866261</td>
    </tr>
    <tr>
      <th>40</th>
      <td>fg_against</td>
      <td>-3.884093</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3pm_for</td>
      <td>-2.002830</td>
    </tr>
    <tr>
      <th>42</th>
      <td>3pm_against</td>
      <td>-1.899977</td>
    </tr>
    <tr>
      <th>43</th>
      <td>fga_for</td>
      <td>-0.055924</td>
    </tr>
    <tr>
      <th>44</th>
      <td>fga_against</td>
      <td>0.048797</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3pa_for</td>
      <td>0.020423</td>
    </tr>
    <tr>
      <th>46</th>
      <td>3pa_against</td>
      <td>-0.010996</td>
    </tr>
    <tr>
      <th>47</th>
      <td>ft_for</td>
      <td>-1.924131</td>
    </tr>
    <tr>
      <th>48</th>
      <td>ft_against</td>
      <td>-1.950374</td>
    </tr>
    <tr>
      <th>49</th>
      <td>fta_for</td>
      <td>-0.034768</td>
    </tr>
    <tr>
      <th>50</th>
      <td>fta_against</td>
      <td>0.020514</td>
    </tr>
    <tr>
      <th>51</th>
      <td>off_rebounds_for</td>
      <td>0.058362</td>
    </tr>
    <tr>
      <th>52</th>
      <td>off_rebounds_against</td>
      <td>-0.049957</td>
    </tr>
    <tr>
      <th>53</th>
      <td>def_rebounds_for</td>
      <td>-0.029409</td>
    </tr>
    <tr>
      <th>54</th>
      <td>def_rebounds_against</td>
      <td>0.028335</td>
    </tr>
    <tr>
      <th>55</th>
      <td>assists_for</td>
      <td>-0.005415</td>
    </tr>
    <tr>
      <th>56</th>
      <td>assists_against</td>
      <td>-0.001259</td>
    </tr>
    <tr>
      <th>57</th>
      <td>steals_for</td>
      <td>0.004033</td>
    </tr>
    <tr>
      <th>58</th>
      <td>steals_against</td>
      <td>-0.003160</td>
    </tr>
    <tr>
      <th>59</th>
      <td>blocks_for</td>
      <td>0.005524</td>
    </tr>
    <tr>
      <th>60</th>
      <td>blocks_against</td>
      <td>0.012153</td>
    </tr>
    <tr>
      <th>61</th>
      <td>turnovers_for</td>
      <td>-0.016100</td>
    </tr>
    <tr>
      <th>62</th>
      <td>turnovers_against</td>
      <td>0.011652</td>
    </tr>
    <tr>
      <th>63</th>
      <td>fouls_for</td>
      <td>0.018368</td>
    </tr>
    <tr>
      <th>64</th>
      <td>fouls_against</td>
      <td>-0.009586</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 2 columns</p>
</div>



The AUC scores are more representative of the model's true performance. This basic logistic model was able to
be accurate 52% of the time with training data and 92% with test data. Therefore, it's a good benchmark, but
there is opportunity to improve the any over-fitting, and through feature engineering (advanced basketball metrics!)

Notice how the AUC score is much better now. With this model (trained on balanced classes), we are much better than a 'no information' classifier. In fact is very close to 1 so it does not display signals of imbalanced classes,
nor overwhelming amount of false positives vs false negatives. We'll revisit this later in model evaluation.

---
# Exploratory Data Analysis

Let us perform EDA to better understand our data. 

## Summary Statistics

### Correlation and Heatmap 

* Final_four is negatively correlated to seed (higher seed -> less likely to be in final four)
* Final_four is positively correlated to fg_for (higher fg_for -> more likely to be in final four)
    * Similar situation for blocks, assists, rebounds, points scored

* Some of these independent variables are highly correlated, so I may not want to use all of them to reduce over-fitting. 
    * For example, points_for is related with points against, so I may want to combine those in a points differential variable, and offensive efficiency will combine (assists_for, fg_for)


```python
quant = df[['final_four', 'season', 'seed', 'team_id', 'points_for',
       'points_against', 'fg_for', 'fg_against', '3pm_for', '3pm_against',
       'fga_for', 'fga_against', '3pa_for', '3pa_against', 'ft_for',
       'ft_against', 'fta_for', 'fta_against', 'off_rebounds_for',
       'off_rebounds_against', 'def_rebounds_for', 'def_rebounds_against',
       'assists_for', 'assists_against', 'steals_for', 'steals_against',
       'blocks_for', 'blocks_against', 'turnovers_for', 'turnovers_against',
       'fouls_for', 'fouls_against']]

stats = df[['points_for','points_against', 'fg_for', 'fg_against', '3pm_for', '3pm_against',
       'fga_for', 'fga_against', '3pa_for', '3pa_against', 'ft_for',
       'ft_against', 'fta_for', 'fta_against', 'off_rebounds_for',
       'off_rebounds_against', 'def_rebounds_for', 'def_rebounds_against',
       'assists_for', 'assists_against', 'steals_for', 'steals_against',
       'blocks_for', 'blocks_against', 'turnovers_for', 'turnovers_against',
       'fouls_for', 'fouls_against']]
```


```python
quant.corr()["final_four"]
```




    final_four              1.000000
    season                 -0.088572
    seed                   -0.671353
    team_id                -0.047393
    points_for              0.420668
    points_against          0.279562
    fg_for                  0.436103
    fg_against              0.299363
    3pm_for                 0.132419
    3pm_against             0.156121
    fga_for                 0.395567
    fga_against             0.374804
    3pa_for                 0.136840
    3pa_against             0.471266
    ft_for                  0.292254
    ft_against             -0.066517
    fta_for                 0.290864
    fta_against            -0.047742
    off_rebounds_for        0.409847
    off_rebounds_against    0.254282
    def_rebounds_for        0.474183
    def_rebounds_against    0.084649
    assists_for             0.432095
    assists_against         0.081193
    steals_for              0.276889
    steals_against          0.158755
    blocks_for              0.366486
    blocks_against          0.141490
    turnovers_for           0.166151
    turnovers_against       0.269311
    fouls_for               0.071425
    fouls_against           0.310043
    Name: final_four, dtype: float64




```python
fig, ax = plt.subplots(figsize=(10, 10))
corr = stats.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, xticklabels = True, yticklabels = True)
```


![png](images/ncaa_85_0.png)



```python
# stats.corr()
```


```python
quant.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>final_four</th>
      <th>season</th>
      <th>seed</th>
      <th>team_id</th>
      <th>points_for</th>
      <th>points_against</th>
      <th>fg_for</th>
      <th>fg_against</th>
      <th>3pm_for</th>
      <th>3pm_against</th>
      <th>...</th>
      <th>assists_for</th>
      <th>assists_against</th>
      <th>steals_for</th>
      <th>steals_against</th>
      <th>blocks_for</th>
      <th>blocks_against</th>
      <th>turnovers_for</th>
      <th>turnovers_against</th>
      <th>fouls_for</th>
      <th>fouls_against</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>...</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
      <td>1840.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.491304</td>
      <td>2009.733152</td>
      <td>5.902717</td>
      <td>1290.800000</td>
      <td>0.082170</td>
      <td>0.070945</td>
      <td>0.028971</td>
      <td>0.025257</td>
      <td>6.917975</td>
      <td>6.258598</td>
      <td>...</td>
      <td>15.264331</td>
      <td>12.244795</td>
      <td>7.338908</td>
      <td>6.283238</td>
      <td>4.388280</td>
      <td>3.265049</td>
      <td>13.204389</td>
      <td>14.603750</td>
      <td>18.038150</td>
      <td>19.742397</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.500060</td>
      <td>4.097770</td>
      <td>4.799378</td>
      <td>96.005048</td>
      <td>0.018478</td>
      <td>0.014926</td>
      <td>0.006705</td>
      <td>0.005426</td>
      <td>1.307789</td>
      <td>0.928820</td>
      <td>...</td>
      <td>2.119950</td>
      <td>1.523999</td>
      <td>1.476626</td>
      <td>0.985864</td>
      <td>1.502522</td>
      <td>0.623705</td>
      <td>1.836439</td>
      <td>2.214709</td>
      <td>2.020666</td>
      <td>1.847527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>2003.000000</td>
      <td>1.000000</td>
      <td>1102.000000</td>
      <td>0.051165</td>
      <td>0.045878</td>
      <td>0.017784</td>
      <td>0.016117</td>
      <td>2.689655</td>
      <td>3.178571</td>
      <td>...</td>
      <td>9.354839</td>
      <td>7.676471</td>
      <td>3.451613</td>
      <td>3.387097</td>
      <td>0.655172</td>
      <td>1.451613</td>
      <td>7.387097</td>
      <td>8.857143</td>
      <td>12.029412</td>
      <td>14.545455</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2006.000000</td>
      <td>2.000000</td>
      <td>1211.750000</td>
      <td>0.069122</td>
      <td>0.060210</td>
      <td>0.024268</td>
      <td>0.021358</td>
      <td>5.970588</td>
      <td>5.633333</td>
      <td>...</td>
      <td>13.812500</td>
      <td>11.170977</td>
      <td>6.330882</td>
      <td>5.616461</td>
      <td>3.300000</td>
      <td>2.812500</td>
      <td>12.029412</td>
      <td>13.090241</td>
      <td>16.771822</td>
      <td>18.468229</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>2010.000000</td>
      <td>4.000000</td>
      <td>1278.500000</td>
      <td>0.077424</td>
      <td>0.067318</td>
      <td>0.027189</td>
      <td>0.023993</td>
      <td>6.875000</td>
      <td>6.234314</td>
      <td>...</td>
      <td>15.200000</td>
      <td>12.156250</td>
      <td>7.183502</td>
      <td>6.241379</td>
      <td>4.107143</td>
      <td>3.214286</td>
      <td>13.190524</td>
      <td>14.343750</td>
      <td>18.062500</td>
      <td>19.646110</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>2013.000000</td>
      <td>10.000000</td>
      <td>1373.250000</td>
      <td>0.090343</td>
      <td>0.078396</td>
      <td>0.031852</td>
      <td>0.027799</td>
      <td>7.774194</td>
      <td>6.838710</td>
      <td>...</td>
      <td>16.608583</td>
      <td>13.176471</td>
      <td>8.133333</td>
      <td>6.900806</td>
      <td>5.261111</td>
      <td>3.645161</td>
      <td>14.299808</td>
      <td>15.831627</td>
      <td>19.243534</td>
      <td>20.847851</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>2017.000000</td>
      <td>16.000000</td>
      <td>1463.000000</td>
      <td>0.197555</td>
      <td>0.155237</td>
      <td>0.068938</td>
      <td>0.055339</td>
      <td>12.148148</td>
      <td>10.468750</td>
      <td>...</td>
      <td>23.958333</td>
      <td>17.625000</td>
      <td>12.500000</td>
      <td>11.041667</td>
      <td>10.962963</td>
      <td>5.793103</td>
      <td>20.875000</td>
      <td>23.708333</td>
      <td>25.111111</td>
      <td>27.916667</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 32 columns</p>
</div>



While the team's season statistics are useful, it may be more predictive to understand a team's performance on a per game basis, to understand how it's winning or losing its games.

The main categorical variable is conferences. It appears to also be predictive from the baseline, representing the strength of the conference. Some conferences are generally very strong and play stronger competition than others.
* big_ten (Michigan State, Purdue, Michigan)
* big_twelve (Texas Tech, Kansas State)
* acc (Duke, UNC, Virgnia)


```python
(pd.DataFrame(regular_season_total.groupby("conference_code")["final_four"].sum()).
sort_values(by="final_four",ascending=False))[0:5]
# See that big_east, acc, big_ten, sec, big_twelve all have 5+ teams that have made it to the final four
# These are the conferences with the best teams
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>final_four</th>
    </tr>
    <tr>
      <th>conference_code</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>big_east</th>
      <td>11</td>
    </tr>
    <tr>
      <th>acc</th>
      <td>10</td>
    </tr>
    <tr>
      <th>big_ten</th>
      <td>10</td>
    </tr>
    <tr>
      <th>sec</th>
      <td>9</td>
    </tr>
    <tr>
      <th>big_twelve</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



---
## Data Visualization


```python
sns.pairplot(df, x_vars = list(quant.columns), y_vars = ["final_four"])
```




    <seaborn.axisgrid.PairGrid at 0x11f8e9710>




![png](images/ncaa_92_1.png)


* Some observations include: 
    * No seeds greater than 11 have made it to final four
    * Teams which make the final four, in the regular season score more, make more FG, give up less FTs, rebound more, block more, foul less, ... all the attributes of a great basketball team

---
# Feature Engineering

## 'Standardize' features

Let's keep the stat metrics on the same scale, on a per game basis rather than standardizing to keep it interpretable. First we need to count the number of games for each team.


```python
# add a column vector to indicate a game was played
regular["game_played"] = np.ones((regular.shape[0], 1))
regular.groupby(["season","winning_team_id"]).sum()
# sum over games where they won
# sum over games where they lost
# sum w + l = total games played

def games_won(team_id, season):
    try: 
        i = regular.groupby(["season","winning_team_id"]).sum().query(f'winning_team_id=={team_id} & season=={season}')["game_played"][0]
    except:
        i = 0
    return i
    
def games_lost(team_id,season):
    try:
        i = regular.groupby(["season","losing_team_id"]).sum().query(f'losing_team_id=={team_id} & season=={season}')["game_played"][0]
    except:
        i = 0
    return i


df["games_won"] = (df[["team_id","season"]].
                         apply(lambda df: 
                               games_won(df["team_id"],
                                         df["season"]),axis=1))

df["games_lost"] = (df[["team_id","season"]].
                             apply(lambda df: 
                                   games_lost(df["team_id"],
                                             df["season"]),axis=1))
```


```python
df["games_total"] = df["games_won"] + df["games_lost"]
```


```python
# Effective gf 
# This statistic adjusts for the fact that a 3-point field goal is worth one more point than a 2-point field goal.
df["efg%"]=(df["fg_for"] + 0.5*df["3pm_for"]) / df["fga_for"]
```


```python
# NTS: time permitting re-factor with FunctionTransformer
df["points_for"] = df["points_for"]/df["games_total"]
df["points_against"] = df["points_against"]/df["games_total"]
df['fg_for'] = df['fg_for']/df["games_total"]
df['fg_against'] = df['fg_against']/df["games_total"]
df['fga_for'] = df['fga_for']/df["games_total"]
df['fga_against'] = df['fga_against']/df["games_total"]
df['3pm_for'] = df['3pm_for']/df["games_total"]
df['3pm_against'] = df['3pm_against']/df["games_total"]
df['3pa_for'] = df['3pa_for']/df["games_total"]
df['3pa_against'] = df['3pa_against']/df["games_total"]
df['ft_for'] = df['ft_for']/df["games_total"]
df['ft_against'] = df['ft_against']/df["games_total"]
df['fta_for'] = df['fta_for']/df["games_total"]
df['fta_against'] = df['fta_against']/df["games_total"]
df['off_rebounds_for'] = df['off_rebounds_for']/df["games_total"]
df['off_rebounds_against'] = df['off_rebounds_against']/df["games_total"]
df['def_rebounds_for'] = df['def_rebounds_for']/df["games_total"]
df['def_rebounds_against'] = df['def_rebounds_against']/df["games_total"]
df['assists_for'] = df['assists_for']/df["games_total"]
df['assists_against'] = df['assists_against']/df["games_total"]
df['steals_for'] = df['steals_for']/df["games_total"]
df['steals_against'] = df['steals_against']/df["games_total"]
df['blocks_for'] = df['blocks_for']/df["games_total"]
df['blocks_against'] = df['blocks_against']/df["games_total"]
df['turnovers_for'] = df['turnovers_for']/df["games_total"]
df['turnovers_against'] = df['turnovers_against']/df["games_total"]
df['fouls_for'] = df['fouls_for']/df["games_total"]
df['fouls_against'] = df['fouls_against']/df["games_total"]
```

## Transform and add new features

Let's use some advanced metrics from the NBA. 
Note, opponent % are kept as to gauge defense as well.


```python
# Now, instead of games_won and games_lost, let's combine this into win_%
df['win_%']=df["games_won"]/df["games_total"]

# +/- point differential aka margin of victory
df["margin_of_victory"]=df["points_for"]-df["points_against"]

# Assist to turnover ratio: this measures your ability to care of possessions and pass the ball, as a team.
df["ast_to_ratio"] = df["assists_for"]/df["turnovers_for"]

# Let's convert field goals, threes and fts into %. Will reduce collinearity and also simplify interpretation.
df['fg%_for'] = df['fg_for']/df['fga_for']
df['fg%_against'] = df['fg_against']/df['fga_against']

df['3p%_for'] = X['3pm_for']/X['3pa_for']
df['3p%_against'] = df['3pm_against']/df['3pa_against']

df['ft%_for'] = df['ft_for']/X['fta_for']
df['ft%_against'] = df['ft_against']/df['fta_against']
```


```python
# drop games won, lost, its redundant with win_%
df = df.drop(["games_won","games_lost"],axis=1) 
```


```python
# 33 games have no data, probably from SMOTE these teams don't exist. Let's just drop for now since I'm getting NaN.
# df[df["win_%"].isna()].index
df = df.drop([1036, 1049, 1134, 1150, 1158, 1187, 1216, 1299, 1310, 1341, 1353,
            1361, 1375, 1389, 1434, 1444, 1452, 1492, 1528, 1530, 1533, 1572,
            1624, 1643, 1653, 1711, 1720, 1749, 1760, 1783, 1850, 1851])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>final_four</th>
      <th>season</th>
      <th>seed</th>
      <th>team_id</th>
      <th>conference_code_a_sun</th>
      <th>conference_code_a_ten</th>
      <th>conference_code_aac</th>
      <th>conference_code_acc</th>
      <th>conference_code_aec</th>
      <th>conference_code_big_east</th>
      <th>...</th>
      <th>efg%</th>
      <th>win_%</th>
      <th>margin_of_victory</th>
      <th>ast_to_ratio</th>
      <th>fg%_for</th>
      <th>fg%_against</th>
      <th>3p%_for</th>
      <th>3p%_against</th>
      <th>ft%_for</th>
      <th>ft%_against</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 77 columns</p>
</div>




```python
df.to_csv("./data/df.csv")
X.shape # 1840 observations
```




    (1872, 65)



I will keep the original features and use regularization methods in the next section.

---
# Modelling
## Logistic Regression with Regularization
### L1: Lasso

I prefer Lasso as a shrinkage method, to narrow down on the most important features, since our baseline model was very accurate already. 


```python
y_test = df[df["season"]>=2015]["final_four"]
y_train = df[df["season"]<2015]["final_four"]
X_train = df.query("season<2015").loc[:,"season":"ft%_against"]
X_test = df.query("season>=2015").loc[:,"season":"ft%_against"]

print(X_train.shape) #85%
print(X_test.shape) #15%

# Instantiate Model
logreg_lasso_1 = LogisticRegression(penalty='l1', C=1)

# Fit model.
logreg_lasso_1.fit(X_train, y_train)

print(f'Logistic Regression Intercept: {logreg_lasso_1.intercept_}')
print(f'Logistic Regression Coefficient: {logreg_lasso_1.coef_}')
print("\n")

# Generate prediction values
print(f'Number of teams making it to final four:{sum(logreg_lasso_1.predict(X_train))}')
print("\n")

# Generate predictions/probabilities
print(logreg_lasso_1.predict_proba(X_train)[:,1])
print("\n")
```

    (1571, 76)
    (269, 76)
    Logistic Regression Intercept: [0.]
    Logistic Regression Coefficient: [[ 0.00141951 -0.77849273 -0.00025749  0.         -5.1456576  -1.70624789
      -4.80742667  0.         -4.83805072  0.          0.         -3.54071377
      -5.4726868   0.          0.         -3.56101615  0.87904808  0.
       0.          0.          0.          0.         -2.57593666 -5.10398929
       0.          0.         -3.1504727  -4.57699273  0.         -3.70504311
       0.          0.          0.          0.          0.         -0.88649024
      -4.78484248  0.          0.          0.          0.         -0.95946952
       0.42807337  0.          0.          0.46165073 -0.20879539  0.28420649
       0.00311739 -0.18050503 -0.18200272  0.16557446 -0.02631732  0.50236311
      -0.45055486 -0.00524669  0.05135045  0.3674497   0.03333877  0.02867236
       0.31202959 -0.25130464  0.03259512  0.3092104  -0.21813342  0.12577743
       0.         -8.48018843  0.          0.          0.          0.
       0.          0.          0.          0.        ]]
    
    
    Number of teams making it to final four:829
    
    
    [0.30300043 0.64610282 0.21053469 ... 0.99991704 0.99990243 0.6140518 ]
    
    



```python
%%html
<style>
table {float:left}
</style>
```


<style>
table {float:left}
</style>



### Interpretation of Coefficients


```python
lasso_coef = pd.DataFrame(list(zip(X_train.columns,logreg_lasso_1.coef_.T[:,0])))
lasso_coef.columns = ["variable","coef"]
lasso_coef["coef_abs"]=abs(lasso_coef["coef"])
lasso_coef.sort_values(by="coef_abs",ascending=False)[0:25]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>coef</th>
      <th>coef_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>win_%</td>
      <td>-8.480188</td>
      <td>8.480188</td>
    </tr>
    <tr>
      <th>12</th>
      <td>conference_code_big_twelve</td>
      <td>-5.472687</td>
      <td>5.472687</td>
    </tr>
    <tr>
      <th>4</th>
      <td>conference_code_a_ten</td>
      <td>-5.145658</td>
      <td>5.145658</td>
    </tr>
    <tr>
      <th>23</th>
      <td>conference_code_mwc</td>
      <td>-5.103989</td>
      <td>5.103989</td>
    </tr>
    <tr>
      <th>8</th>
      <td>conference_code_big_east</td>
      <td>-4.838051</td>
      <td>4.838051</td>
    </tr>
    <tr>
      <th>6</th>
      <td>conference_code_acc</td>
      <td>-4.807427</td>
      <td>4.807427</td>
    </tr>
    <tr>
      <th>36</th>
      <td>conference_code_wcc</td>
      <td>-4.784842</td>
      <td>4.784842</td>
    </tr>
    <tr>
      <th>27</th>
      <td>conference_code_pac_twelve</td>
      <td>-4.576993</td>
      <td>4.576993</td>
    </tr>
    <tr>
      <th>29</th>
      <td>conference_code_sec</td>
      <td>-3.705043</td>
      <td>3.705043</td>
    </tr>
    <tr>
      <th>15</th>
      <td>conference_code_cusa</td>
      <td>-3.561016</td>
      <td>3.561016</td>
    </tr>
    <tr>
      <th>11</th>
      <td>conference_code_big_ten</td>
      <td>-3.540714</td>
      <td>3.540714</td>
    </tr>
    <tr>
      <th>26</th>
      <td>conference_code_pac_ten</td>
      <td>-3.150473</td>
      <td>3.150473</td>
    </tr>
    <tr>
      <th>22</th>
      <td>conference_code_mvc</td>
      <td>-2.575937</td>
      <td>2.575937</td>
    </tr>
    <tr>
      <th>5</th>
      <td>conference_code_aac</td>
      <td>-1.706248</td>
      <td>1.706248</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3pm_for</td>
      <td>-0.959470</td>
      <td>0.959470</td>
    </tr>
    <tr>
      <th>35</th>
      <td>conference_code_wac</td>
      <td>-0.886490</td>
      <td>0.886490</td>
    </tr>
    <tr>
      <th>16</th>
      <td>conference_code_horizon</td>
      <td>0.879048</td>
      <td>0.879048</td>
    </tr>
    <tr>
      <th>1</th>
      <td>seed</td>
      <td>-0.778493</td>
      <td>0.778493</td>
    </tr>
    <tr>
      <th>53</th>
      <td>def_rebounds_for</td>
      <td>0.502363</td>
      <td>0.502363</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3pa_for</td>
      <td>0.461651</td>
      <td>0.461651</td>
    </tr>
    <tr>
      <th>54</th>
      <td>def_rebounds_against</td>
      <td>-0.450555</td>
      <td>0.450555</td>
    </tr>
    <tr>
      <th>42</th>
      <td>3pm_against</td>
      <td>0.428073</td>
      <td>0.428073</td>
    </tr>
    <tr>
      <th>57</th>
      <td>steals_for</td>
      <td>0.367450</td>
      <td>0.367450</td>
    </tr>
    <tr>
      <th>60</th>
      <td>blocks_against</td>
      <td>0.312030</td>
      <td>0.312030</td>
    </tr>
    <tr>
      <th>63</th>
      <td>fouls_for</td>
      <td>0.309210</td>
      <td>0.309210</td>
    </tr>
  </tbody>
</table>
</div>



* Win_% is the most important predictor. Although it's strange to see that it has a negative sign, this is probably the case because schools that make to to the final four actually have lower win % than schools who do not make it to the final four because they face tougher competition (57% avg vs 72% avg). This supports the argument that which conference you play for is very important in predictive power. 

* Again, your strongest conferences, are the big_twelve, acc, big_ten. If you play in acc, your likelihood of reaching final four is very low, because you will likely be dominated by the top tier schools in those conferences. 

* In terms of regular season states, 3PM for and against is important, as the game continues to move shooting more threes, as it's been proven to be an effective strategy. 

* Lastly, seed is not to be overlooked as it's a good composite score for a team's strength, capturing a lot of regular season performance. As your seed increases by 1, your likelihood of reaching final four is 0.47x less. This makes sense... a 3 seed only has a ~10% (0.47^3) of making to final four! This is consistent with FiveThirtyEight's forecasts. Purdue, Houston, & Texas Tech had 10-14% of making it to the Final four.

* Similar to the NBA, margin of victory is a predictive variable though with a positive correlation. In the NBA, there is more parity since each team plays one another at least twice in a season, so individual team stats are more important than seed & conference.

---
## Model Evaluation


```python
# Score the model - accuracy
print(f' Logreg train accuracy: {cross_val_score(logreg_lasso_1, X_train, y_train, cv=5).mean()}')
print("\n")

print(f' Logreg test accuracy: {cross_val_score(logreg_lasso_1, X_test, y_test, cv=5).mean()}')
print("\n")

# Area under the curve
print(f' Area under the curve: {roc_auc_score(y_test, logreg_lasso_1.predict_proba(X_test)[:,1])}')
```

    /Users/gc/anaconda3/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


     Logreg train accuracy: 0.9185457783772837
    
    
     Logreg test accuracy: 0.9073070325900513
    
    
     Area under the curve: 0.9681412337662337


|Metric              | Baseline | Logistic Regression, Lasso = 1 |
|--------------------| -------- | ------------------------------ |
|Train Accuracy      | 0.915    | 0.921                          |
|Test Accuracy       | 0.914    | 0.903                          |
|Area under the curve| 0.940    | 0.961                          |

Even with many fewer variables, this model is arguably as performant as the original.
It does not overfit, and even improves the AUC score by 2% points.


```python
# create data frame of true values and predicted probabilities on test set
pred_proba = [i[1] for i in logreg_lasso_1.predict_proba(X_test)]

pred_df = pd.DataFrame({'true_values': y_test,
                        'pred_probs':pred_proba})
```


```python
# Create figure.
plt.figure(figsize = (8,8))

# Create threshold values.
thresholds = np.linspace(0, 1, 200)

# Define function to calculate sensitivity. (True positive rate.)
def TPR(df, true_col, pred_prob_col, threshold):
    true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
    false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]
    return true_positive / (true_positive + false_negative)
    

# Define function to calculate 1 - specificity. (False positive rate.)
def FPR(df, true_col, pred_prob_col, threshold):
    true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
    false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]
    return 1 - (true_negative / (true_negative + false_positive))
    
# Calculate sensitivity & 1-specificity for each threshold between 0 and 1.
tpr_values = [TPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
fpr_values = [FPR(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]

# Plot ROC curve.
plt.plot(fpr_values, # False Positive Rate on X-axis
         tpr_values, # True Positive Rate on Y-axis
         label='ROC Curve')

# Plot baseline. (Perfect overlap between the two populations.)
plt.plot(np.linspace(0, 1, 200),
         np.linspace(0, 1, 200),
         label='baseline',
         linestyle='--')

# Label axes.
plt.title(f'ROC Curve with AUC = {round(roc_auc_score(pred_df["true_values"], pred_df["pred_probs"]),2)}', fontsize=20)
plt.ylabel('Sensitivity', fontsize=15)
plt.xlabel('1 - Specificity', fontsize=15)

# Create legend.
plt.legend(fontsize=16);
```


![png](images/ncaa_118_0.png)



```python
cm = confusion_matrix(y_test, logreg_lasso_1.predict(X_test))
cm_df = pd.DataFrame(data=cm, columns=['predicted negative', 'predicted positive'], index=['actual negative', 'actual positive'])
cm_df
# more false negatives than false positives
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted negative</th>
      <th>predicted positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actual negative</th>
      <td>183</td>
      <td>9</td>
    </tr>
    <tr>
      <th>actual positive</th>
      <td>7</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>



## Hyperparameter Selection 
Let's fine tune our hyperparameter of alpha with GridSearchCV to see if we can improve our model further.


```python
log_params = {
    'penalty':["l1", "l2"],
    'C':list(np.linspace(0.01, 5, 10))
}

log_gridsearch = GridSearchCV(LogisticRegression(random_state=8), log_params, cv=5, verbose=1, n_jobs=2)

log_gridsearch = log_gridsearch.fit(X_train, y_train)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits


    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:   14.6s
    [Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:   33.5s finished



```python
print(log_gridsearch.best_score_)
log_gridsearch.best_params_
```

    0.9325270528325907





    {'C': 5.0, 'penalty': 'l1'}




```python
log_params = {
    'penalty':["l1"],
    'C':list(np.linspace(0.01, 5, 10))
}

log_gridsearch = GridSearchCV(LogisticRegression(random_state=8), log_params, cv=5, verbose=1, n_jobs=2)

log_gridsearch = log_gridsearch.fit(X_train, y_train)
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits


    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done  50 out of  50 | elapsed:   26.1s finished



```python
plt.figure(figsize = (5,5))

lst_of_c = [c["C"] for c in pd.DataFrame(log_gridsearch.cv_results_)["params"]]
mean_test_scores = pd.DataFrame(log_gridsearch.cv_results_)["mean_test_score"]
plt.plot(lst_of_c, 
         mean_test_scores)
plt.title(f'Penalty Strength vs Test Accuracy: {round(log_gridsearch.best_score_,3)}', fontsize=15)
plt.ylabel('C', fontsize=10)
plt.xlabel('%', fontsize=10)
```




    Text(0.5, 0, '%')




![png](images/ncaa_124_1.png)



```python
print(log_gridsearch.best_score_)
log_gridsearch.best_params_
```

    0.9325270528325907





    {'C': 5.0, 'penalty': 'l1'}



## Best Model


```python
# Instantiate Model
logreg_best = LogisticRegression(penalty='l1', C=log_gridsearch.best_params_["C"])

# Fit model.
logreg_best.fit(X_train, y_train)

# Generate prediction values
yhat_train = logreg_lasso_1.predict(X_train)
yhat_test = logreg_lasso_1.predict(X_test)
print("\n")

# Generate predictions/probabilities
yhat_train_proba = logreg_best.predict_proba(X_train)[:,1]
yhat_test_proba = logreg_best.predict_proba(X_test)[:,1]

# Score the model - accuracy
print(f' Logreg train accuracy: {cross_val_score(logreg_best, X_train, y_train, cv=5).mean()}')
print("\n")

print(f' Logreg test accuracy: {cross_val_score(logreg_best, X_test, y_test, cv=5).mean()}')
print("\n")

# Area under the curve
print(f' Area under the curve: {roc_auc_score(y_test, logreg_best.predict_proba(X_test)[:,1])}')
```

    
    
     Logreg train accuracy: 0.9319095069161205
    
    


    /Users/gc/anaconda3/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


     Logreg test accuracy: 0.9187650085763295
    
    
     Area under the curve: 0.9662472943722944


    /Users/gc/anaconda3/lib/python3.6/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


|Metric              | Baseline | Logistic Regression, Lasso = 1 | Logistic Regression, Lasso = Optimal |
|--------------------| -------- | ------------------------------ | -------------------------------------|
|Train Accuracy      | 0.915    | 0.921                          | 0.936                                |
|Test Accuracy       | 0.914    | 0.903                          | 0.918                                |
|Area under the curve| 0.940    | 0.961                          | 0.966                                |


```python
pred_15_17 = pd.DataFrame(zip(X_test[["season","team_id"]],yhat_test))
pred_15_17 = pd.DataFrame(
    {"season": X_test["season"],
     "team_id": X_test["team_id"],
     "final_four": y_test,
     "yhat": yhat_test  
    }
)
```


```python
results = pd.merge(pred_15_17,teams, how="left",on=["season","team_id"]).drop_duplicates().reset_index()
results.sort_values(by="final_four",ascending=False).drop("index",axis=1)
results.head()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-274-5393da1d7061> in <module>
    ----> 1 results = pd.merge(pred_15_17,teams, how="left",on=["season","team_id"]).drop_duplicates().reset_index()
          2 results.sort_values(by="final_four",ascending=False).drop("index",axis=1)
          3 results.head()


    NameError: name 'pred_15_17' is not defined


---
# Results 

Overall, I was able to improve from the baseline logistic model, by 1.5% in train accuracy, 0% in test accuracy, and 2.6% in ROC curve. What I am impressed about is that the best model via Lasso is able to predict with less variables, while also maintaining interpretability (no wild transformations). 

In summary:

* Win_% is the most important predictor. Although it's strange to see that it has a negative sign, this is probably the case because schools that make to to the final four actually have lower win % than schools who do not make it to the final four because they face tougher competition (57% avg vs 72% avg). This supports the argument that which conference you play for is very important in predictive power. 

* Again, your strongest conferences, are the big_twelve, acc, big_ten. If you play in acc, your likelihood of reaching final four is very low, because you will likely be dominated by the top tier schools in those conferences. 

* In terms of regular season states, 3PM for and against is important, as the game continues to move shooting more threes, as it's been proven to be an effective strategy. 

* Lastly, seed is not to be overlooked as it's a good composite score for a team's strength, capturing a lot of regular season performance. As your seed increases by 1, your likelihood of reaching final four is 0.47x less. This makes sense... a 3 seed only has a ~10% (0.47^3) of making to final four! This is consistent with FiveThirtyEight's forecasts. Purdue, Houston, & Texas Tech had 10-14% of making it to the Final four.

* Similar to the NBA, margin of victory is a predictive variable though with a positive correlation. In the NBA, there is more parity since each team plays one another at least twice in a season, so individual team stats are more important than seed & conference.

My model predicts the probability of a team reaching the final four, primarily based on its regular season data, seed, and conference. I would recommend coaches to develop the three ball ability, as it continues dominate in the NBA and at the college level, and to outperform in the regular season to improve your seeding, to ultimately improve your chances in moving to the final four, through playing weaker opponents. 

Assumptions 
* In this model, there are things that we cannot predict, including injuries, player specific data, draft prospects, travel & time between games (are players are well rested?), coaches track record, momentum of a team firing on all cylinders (last 10 games), given the absence of data.

* In this example, there isn't a penalty on false positives vs false negatives, as wrong is wrong. Predicting a loser who won isn't worse than a winner that lost. There may be implications in gambling, but that's out of scope. 

Research Links
* https://projects.fivethirtyeight.com/2019-march-madness-predictions/ 

---
##  Next Steps

This was a very fun exploration and may plan to revisit this project in a future blog post.

Ideas:
* Simulating winner between 2 games (or at the very least, the final four), using a Poisson model for pts scored
* Logistic vs Non Parametric (KNN, SVM, random forests)
* Scrap 2018 data, and run model, and add graphs (with SMOTE, 2015->2017 data is obfuscated)
* Feature engineer strength of schedule (they have this on sports reference)
* ELO
* No. of previous appearances in history
* Confidence intervals of predicted probabilities