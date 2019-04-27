Title: Lambda function for data wrangling
Date: 2019-03-28 3:00
Tags: python
Slug: lambda

Coming from a math background, to me, Lambda's represented the parameter in exponential distributions, eigenvalues in linear algebra, or in the context of AWS Lambda, a compute service for managing servers!
 

Whereas in programming, by definition, Lambda is an *anonymous* function. This means that unlike normal functions definitions, Lambdas do not have a name.  Often, it's used as an argument for a higher-order function, and is lightweight in syntax, allowing your code to be less verbose, and easier to understand. 


For me, I choose to write a Lambda function over a named function, if it's a simple function that will only be used a few times, passed as an argument. 
<br>

### Lambda functions in Python
---

Here's a simple function:

```python
def hello(name):
    return f' Hello, my name is {name}'
```

And here is its `lambda` equivalent:

```python
lambda name: f' Hello, my name is {name}'
```

The primary differences between named and lambda functions are:

1. `lambda` functions don't have a name

2. `lambda` functions are written on one line

3. `lambda` functions don't require a `return`. It's implied.
<br>

### Syntax
---

`lambda  `   `arguments:  ` `expression` 

Lambda functions can be used wherever function objects are required. A lambda function can take any number of arguments, but can only have one expression.  They can also be used anywhere ordinary functions can! 
<br>

### Example 1
---

`lambda` is powerful when used in conjunction with a pandas data frame and `apply`.  Given that we're in the Sweet Sixteen of March madness, let's use NCAA data.
<br>


```python
import pandas as pd
from IPython.display import HTML

df = pd.DataFrame({
    'college': ['Duke', 'North Carolina', 'Virginia', 'Oregon'],
    'seed': [1, 1, 1, 12],
    'final_four_%': [55, 38, 53, 3]
    })
```


```python
HTML(df.to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>college</th>
      <th>seed</th>
      <th>final_four_%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Duke</td>
      <td>1</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North Carolina</td>
      <td>1</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Virginia</td>
      <td>1</td>
      <td>53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Oregon</td>
      <td>12</td>
      <td>3</td>
    </tr>
  </tbody>
</table>



Let's add a new column, *upset*, identifying whether or not a college is the favourite or underdog, using a ```lambda``` function.

With ```lambda``` , we're able to quickly spin up a function that looks up if a seed is a favourite or an underdog. 
<br>


```python
df["upset"] = df.seed.apply(lambda x, lookup = {1: "Favourite", 12: "Underdog"} : lookup[x])
```


```python
# Sort by probability of reaching final four
HTML(df.sort_values("final_four_%",ascending=False).to_html(classes="table table-repsponsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-repsponsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>college</th>
      <th>seed</th>
      <th>final_four_%</th>
      <th>upset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Duke</td>
      <td>1</td>
      <td>55</td>
      <td>Favourite</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Virginia</td>
      <td>1</td>
      <td>53</td>
      <td>Favourite</td>
    </tr>
    <tr>
      <th>1</th>
      <td>North Carolina</td>
      <td>1</td>
      <td>38</td>
      <td>Favourite</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Oregon</td>
      <td>12</td>
      <td>3</td>
      <td>Underdog</td>
    </tr>
  </tbody>
</table>



### Example 2
---
Here's a sorting example.


```python
prospects = [
    {'name':'Zion Williamson', 'weight': 280, 'rank': 1}, 
    {'name': 'Ja Morant', 'weight': 175, 'rank':2},
    {'name': 'RJ Barrett', 'weight': 210, 'rank':3},
    {'name':'Bol Bol', 'weight': 235, 'rank':18}]
```

Let's figure out who the lightest prospect is, using ```sorted``` a Python method.


```python
sorted(prospects, key= lambda x: x['weight'])

lightest_prospect = sorted(prospects, key= lambda x: x['weight'])[0]["name"]

print(lightest_prospect)
```

    Ja Morant


With ```lambda``` in our toolbelt, we'll have a function to write elegant and beautiful code for data wrangling. 
<br>