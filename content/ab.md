Title: Are the Warriors better without Kevin Durant? 
Date: 2019-06-10 12:00
Topic: Bayesian A/B Test
Slug: ab

In the media, there have been debates about whether or not the Golden State Warriors (GSW) are better without Kevin Durant (KD). From the eye-test, it's laughable to even suggest this, as he's one of the top 3 players in the league (Lebron, KD, Kawhi). Nonetheless, people argue that ball movement is better without him, and therefore make the GSW more lethal.

But, just because the Warriors won a title without KD, that does not mean they don't need him more than ever. At the time of writing, the Toronto Raptors lead 3-1 in the Finals! #WeTheNorth 🦖🍁

Using Bayesian estimation, we can A/B test this hypothesis, by comparing two treatment groups: games played with KD vs. without KD.

Bayesian statistics are an excellent tool to reach for when sample sizes are small, as we can introduce explicit assumptions into the model, when there aren't thousands of observations. 

---

### Primer on Bayesian Statistics

<img src="images/dists.png" class="img-responsive">

$$P\left(model\;|\;data\right) = \frac{P\left(data\;|\;model\right)}{P(data)}\; P\left(model\right)$$


$$ \text{prior} = P\left(model\right) $$
> The **prior** is our belief in the model given no additional information. In our example, this is the mean win % with KD playing.
<br>

$$ \text{likelihood} = P\left(data\;|\;model\right) $$
> The **likelihood** is the probability of the data we observed occurring given the model.
<br>

$$ \text{marginal probability of data} = P(data) $$
> The **marginal probability** of the data is the probability that our data are observed regardless of what model we choose or believe in. 
<br>

$$ \text{posterior} = P\left(model\;|\;data\right) $$
> The **posterior** is our _updated_ belief in the model given the new data we have observed. Bayesian statistics are all about updating a prior belief we have about the world with new data, so we're transforming our _prior_ belief into this new _posterior_ belief about the world. <br><br> In this example, this is the GSW mean winning % with KD playing, given the game logs from the past three seasons.

<br> Note, a Bayesian approach is different from a Frequentist's. Rather than only testing whether two groups are different, we instead pursue an estimate of _how_ different they are, from the posterior distribution.

#### Objective
To calculate the distribution of the posterior probability of GSW mean winning % with KD and without KD.
Moreover, we can calculate the _delta_ between both probabilities to determine if the mean is statistically different from zero (i.e. no difference with or without him).

---

#### Observed Data


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import pymc3 as pm
from IPython.display import HTML
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
from IPython.core.pylabtools import figsize
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

As the competition is much higher in the playoffs, let's analyze playoff vs. regular Season data separately. We can run one test on the regular season, and one test for the playoffs.  

Data is from [Basketball Reference](https://www.basketball-reference.com/).

---
### Regular Season

<table class="table-responsive table-striped  table-bordered">
 <thead>
    <tr>
      <th scope="col">Regular Season</th>
      <th scope="col">With Kevin Durant</th>
      <th scope="col">No Kevin Durant</th>
      <th scope="col">Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2019</td>
      <td>0.69 <br> {'W': 54, 'L': 24} </td>
      <td>0.75 <br> {'W': 3, 'L': 1} </td>
      <td>Record is better when KD is out, but small sample size.</td>
    </tr>
    <tr>
      <td>2018</td>
      <td>0.72 <br> {'W': 49, 'L': 19} </td>
      <td>0.64 <br> {'W': 9, 'L': 5} </td>
      <td>Record is better when KD plays</td>
    </tr>
    <tr>
      <td>2017</td>
      <td>0.82 <br> {'W': 51, 'L': 11} </td>
      <td>0.80 <br> {'W': 16, 'L': 4} </td>
      <td>Record is better when KD plays</td>
    </tr>
    <tr>
      <td>Total (3 seasons)</td>
      <td>0.740 <br> {'W': 154, 'L': 54} </td>
      <td>0.737 <br> {'W': 28, 'L': 10} </td>
      <td>Record is better when KD plays</td>
    </tr>
  </tbody>
</table>

Over the last three seasons with the Warriors, KD has missed 38 games regular season games, and played in 208.


```python
def occurrences(year, kd=True):
    '''occurences(2019, kd=True)
    By default, kd=True means with KD healthy'''
    # clean data
    # regular season
    data = pd.read_csv(f'./data/ab/{year}.txt', sep=',')
    new_columns = ['Rk', 'G', 'Date', 'Age', 'Tm', 'Away', 'Opp', 'Result', 'GS',
       'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB',
       'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'GmSc', '+/-']
    data.columns=new_columns
    # replace did not dress with inactive
    data.GS = np.where(data.GS == 'Did Not Dress','Inactive',data.GS)
    if kd == False:
        game_logs = list(data[data.GS=='Inactive'].Result)
    else:
        game_logs = list(data[data.GS!='Inactive'].Result)
    results = [game.split(' ')[0] for game in game_logs]
    occurrences = [1 if result == 'W' else 0 for result in results]
    return occurrences
```


```python
regular_season_with_kd = occurrences(2019, kd=True)+occurrences(2018, kd=True)+occurrences(2017, kd=True)
regular_season_no_kd = occurrences(2019, kd=False)+occurrences(2018, kd=False)+occurrences(2017, kd=False)
print(f'Observed win % when Kevin Durant plays: {round(np.mean(regular_season_with_kd),4)}')
print(f'Observed win % when Kevin Durant does not play: {round(np.mean(regular_season_no_kd),4)}')
```

    Observed win % when Kevin Durant plays: 0.7404
    Observed win % when Kevin Durant does not play: 0.7368


* Note, we do not know the true win %, only the observed win %. We infer the true quantity from the observed data.

* Notice the unequal sample sizes (208 vs. 38), but this is not problem in Bayesian analysis. We will see the uncertainty of the smaller sample size captured in the posterior distribution. 

---
#### Bayesian Tests with MCMC

* Markov Chain Monte Carlo (MCMC) is a method to find the posterior distribution of our parameter of interest.
> This type of algorithm generates Monte Carlo simulations in a way that relies on the Markov property, then accepts these simulations at a certain rate to get the posterior distribution.

* We will use [PyMC3](https://docs.pymc.io/), a probabilistic library for Python to generate MC simulations.

* Before seeing any of the data, my prior is that GSW will win between 50% - 90% of their games, because they are an above average basketball team, and no team has ever won more than 72 games.


```python
# Instantiate
observations_A = regular_season_with_kd
observations_B = regular_season_no_kd

with pm.Model() as model:
    # Assume Uniform priors for p_A and p_B    
    p_A = pm.Uniform("p_A", 0.5, .9)
    p_B = pm.Uniform("p_B", 0.5, .9)
    
    # Define the deterministic delta function. This is our unknown of interest.
    # Delta is deterministic, no uncertainty beyond p_A and p_B
    delta = pm.Deterministic("delta", p_A - p_B)
    
    # We have two observation datasets: A, B
    # Posterior distribution is Bernoulli
    obs_A = pm.Bernoulli("obs_A", p_A, observed=observations_A)
    obs_B = pm.Bernoulli("obs_B", p_B, observed=observations_B)
    
    # Draw samples from the posterior distribution
    trace = pm.sample(20000)
    burned_trace=trace[1000:]
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [p_B, p_A]
    Sampling 2 chains: 100%|██████████| 41000/41000 [00:21<00:00, 1899.05draws/s]


* Using PyMC3, we generated a trace, or chain of values from the posterior distribution 
* Generated 20,000 samples from the posterior distribution (20,000 samples / chain / core)

Because this algorithm needs to converge, we set a number of tuning steps (1,000) to occur first and where the algorithm should "start exploring." It's good to see the Markov Chains overlap, which suggests convergence.

<img src="images/trace.svg" class="img-responsive">


```python
df = pm.summary(burned_trace).round(2)[['mean', 'sd', 'hpd_2.5', 'hpd_97.5']]
HTML(df.to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p_A</th>
      <td>0.74</td>
      <td>0.03</td>
      <td>0.68</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>p_B</th>
      <td>0.73</td>
      <td>0.07</td>
      <td>0.59</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>delta</th>
      <td>0.01</td>
      <td>0.07</td>
      <td>-0.13</td>
      <td>0.16</td>
    </tr>
  </tbody>
</table>



* Unlike with confidence intervals (frequentist), there is a measure of probability with the credible interval.
* There is a 95% probability that the true win rate with KD is in the interval (0.68, 0.79).
* There is a 95% probability that the true win rate with no KD is in the interval (0.59, 0.85).

<img src="images/reg_season.svg" class="img-responsive">

Note, the 2.5% and 97.5% markers indicate the quantiles for the credible interval, similar to the confidence interval in frequentist statistics.

---
#### Results

* In the third graph, the posterior win rate is 1.2% higher when KD plays in the regular season.

* Observe that because have less data for when KD is out, our posterior distribution of  $𝑝_𝐵$ is wider, implying we are less certain about the true value of $𝑝_𝐵$ than we are of $𝑝_𝐴$. The 95% credible interval is much wider for $𝑝_𝐵$, as there is a smaller sample size, for when KD did not play. We are less certain that the GSW wins 73% of the time without KD.

* The difference in sample sizes ($N_B$ < $N_A$) naturally fits into Bayesian analysis, whereas you need the same populations for frequentist approach!


```python
# Count the number of samples less than 0, i.e. the area under the curve
print("Probability that GSW is worse with Kevin Durant in the regular season: %.2f" % \
    np.mean(delta_samples < 0))

print("Probability that GSW is better with Kevin Durant in the regular season: %.2f" % \
    np.mean(delta_samples > 0))
```

    Probability that GSW is worse with Kevin Durant in the regular season: 0.45
    Probability that GSW is better with Kevin Durant in the regular season: 0.55


The probabilities are pretty close, so we can chalk this up to the Warriors having a experienced supporting cast. 

There is significant overlap between the distribution pf posterior $p_A$ and posterior of $p_B$, so one is not better than the other with high probability. The majority of the distribution of delta is around 0, so there is no statistically difference between the groups in the regular season.

Ideally, we should perform more trials when KD is injured (as each data point for scenario B contributes more inferential power than each additional point for scenario A). One could do a similar analysis for when he played on the Oklahoma City Thunder.

---
### Playoffs
#### Do superstars shine when the stakes are highest?

<table class="table-responsive table-striped  table-bordered">
 <thead>
    <tr>
      <th scope="col">Playoffs</th>
      <th scope="col">With Kevin Durant</th>
      <th scope="col">No Kevin Durant</th>
      <th scope="col">Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2019</td>
      <td>0.64 <br> {'W': 7, 'L': 4} </td>
      <td>0.66 <br> {'W': 6, 'L': 3} </td>
      <td>Record is marginally better when KD is out, but small sample size. Skewed by Portland series, which GSW won 4-0 with KD injured.</td>
    </tr>
    <tr>
      <td>2018</td>
      <td>0.76 <br> {'W': 16, 'L': 5} </td>
      <td>n/a <br> {'W': 0, 'L': 0} </td>
      <td>KD did not miss any games. Won Championship.</td>
    </tr>
    <tr>
      <td>2017</td>
      <td>0.82 <br> {'W': 14, 'L': 1} </td>
      <td>1 <br> {'W': 2, 'L': 0}. Small sample size. </td>
      <td>Won championship.</td>
    </tr>
      <td>Total (3 seasons)</td>
      <td>0.79 <br> {'W': 37, 'L': 10} </td>
      <td>0.73 <br> {'W': 8, 'L': 3} </td>
      <td>Record is better when KD plays</td>
  </tbody>
</table>


```python
playoffs_with_kd = occurrences('2019_playoffs', kd=True)+occurrences('2018_playoffs', kd=True)+occurrences('2017_playoffs', kd=True)
playoffs_no_kd = occurrences('2019_playoffs', kd=False)+occurrences('2018_playoffs', kd=False)+occurrences('2017_playoffs', kd=False)
print(f'Observed win % when Kevin Durant plays: {round(np.mean(playoffs_with_kd),2)}')
print(f'Observed win % when Kevin Durant does not play: {round(np.mean(playoffs_no_kd),2)}')
```

    Observed win % when Kevin Durant plays: 0.79
    Observed win % when Kevin Durant does not play: 0.73


Over the last three playoff runs with the Warriors, KD has missed 11, and played in 47. By combining results from the past three seasons, we obtain a larger test group, which allows us to observe a real change vs. looking at the statistics for a single year. The difference is more pronounced across three seasons. 

Let's simulate to see investigate if GSW has a higher win % with KD in the playoffs. 


```python
playoff_obs_A = playoffs_with_kd
playoff_obs_B = playoffs_no_kd

with pm.Model() as playoff_model:
    playoff_p_A = pm.Uniform("playoff_p_A", 0, 1)
    playoff_p_B = pm.Uniform("playoff_p_B", 0, 1)
    
    playoff_delta = pm.Deterministic("playoff_delta", playoff_p_A - playoff_p_B)
    
    playoff_obs_A = pm.Bernoulli("playoff_obs_A", playoff_p_A, observed=playoff_obs_A)
    playoff_obs_B = pm.Bernoulli("playoff_obs_B", playoff_p_B, observed=playoff_obs_B)

    playoff_trace = pm.sample(20000)
    playoff_burned_trace=playoff_trace[1000:]
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [playoff_p_B, playoff_p_A]
    Sampling 2 chains: 100%|██████████| 41000/41000 [00:23<00:00, 1709.99draws/s]



```python
df2 = pm.summary(playoff_burned_trace).round(2)[['mean', 'sd', 'hpd_2.5', 'hpd_97.5']]
HTML(df2.to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>playoff_p_A</th>
      <td>0.78</td>
      <td>0.06</td>
      <td>0.66</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>playoff_p_B</th>
      <td>0.69</td>
      <td>0.12</td>
      <td>0.46</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>playoff_delta</th>
      <td>0.08</td>
      <td>0.14</td>
      <td>-0.17</td>
      <td>0.36</td>
    </tr>
  </tbody>
</table>



<img src="images/playoffs.svg" class="img-responsive">


```python
# Count the number of samples less than 0, i.e. the area under the curve
print("Probability that GSW is worse with Kevin Durant in the playoffs: %.2f" % \
    np.mean(playoff_delta_samples < 0))

print("Probability that GSW is better with Kevin Durant in the playoffs: %.2f" % \
    np.mean(playoff_delta_samples > 0))
```

    Probability that GSW is worse with Kevin Durant in the playoffs: 0.28
    Probability that GSW is better with Kevin Durant in the playoffs: 0.72


---
#### Are the Warriors better without Kevin Durant? 

No.

We can see that while delta = 0 (i.e. no effect when KD plays) is in the credible interval at 95%, the majority of the distribution is above 0. This A/B test implies that the treatment group with KD, is likely better than the group without KD. In fact, the probability that GSW is better with Kevin Durant in the playoffs is 72%, a significant jump from 55% in the regular season! 

Superstars make a difference in the playoffs. The regular season is where you make your name, but the postseason is where you make your fame. The delta is 8% higher with KD. That's the advantage you gain with a player of his caliber, as he can hit clutch shots when it matters most.

#### References
* [Significant Samples](https://multithreaded.stitchfix.com/blog/2015/05/26/significant-sample/)
* [May Bayes Theorem Be With You](https://multithreaded.stitchfix.com/blog/2015/02/12/may-bayes-theorem-be-with-you/)