Title: Can machines pick up sarcasm?
Date: 2019-05-12 12:00
Topic: Natural Language Processing
Slug: reddit

What's the difference between a life pro tip, and one that is a bit more questionable? This can be sometimes a subtle difference, or a moral grey area to distinguish, even for humans.  

Life Pro Tip: 
> A concise and specific tip that improves life for you and those around you in a specific and significant way.

<br>

_Example_: "If you want to learn a new language, figure out the 100 most frequently used words and start with them. Those words make up about 50% of everyday speech, and should be a very solid basis."

> An Unethical Life Pro Tip is a tip that improves your life in a meaningful way, perhaps at the expense of others and/or with questionable legality. Due to their nature, do not actually follow any of these tips–they're just for fun. 

<br>

Example: "Save business cards of people you don't like. If you ever hit a parked car accidentally, just write "sorry" on the back and leave it on the windshield."

Let's collect posts (web scrap) from 2 subreddits, and create a machine learning model using Natural Langauge Processing (NLP) to classify which subreddit a particular post belongs too. Can my model pick up on sarcasm, internet 'trolling', or tongue-in-cheek semantic of sentences? Probably not, but let's try. I hope you have as much fun playing with this, as I did making it. 

If you're feeling lucky, visit my app for a [Life Pro Tip](https://nlp-reddit-garry.herokuapp.com)!

---

### Reddit API 

Fortunately, Reddit provides a public JSON end point, so we can easily consume that format, and manipulate it a Pandas DataFrame. Simply add `.json` at the end of the URL.

If you plan to run your own `get` requests, keep in mind that Reddit has a limit of 25 posts / request. In conjunction with `for` loop, write a `time.sleep()` function in Python (or something equivalent) to avoid a 429 Too Many Requests error. 



#### Data dictionary

We are interested in the following features:

<table class="table table-responsive table-bordered">
<thead>
</thead>
<tbody>
<tr>
<td><b>Target variable, y </b></td>
<td> <p> subreddit (str) </p> </td>
</tr>
    
<tr>
<td><b>Design matrix, X </b></td>
<td> 
<p>title (str)</p>
<p>score (int)</p>
<p>num_comments (int)</p>
<p>author (int)</p>
<p>name (int)</p>
</ul>
</td></tr>   

</tbody>
</table>

--- 
### Pre-processing data


First, I have to pre-process the data, and use natural language processing packages to tokenize strings to individual words. We will be using Python's Natural Language Toolkit (nltk) package.

Follow along if you want to create your own classifier, otherwise, skip to the results. All code is available on [GitHub](https://github.com/garrrychan/nlp_reddit). Refer to `reddit_garry.py` for scraping.


```python
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning) 
np.set_printoptions(suppress=True)
pd.set_option('display.max_colwidth', -1)
from IPython.display import HTML
```


```python
# encoding utf-8 for special characters
raw_lpt = pd.read_csv("./data/lpt.csv", encoding='utf-8')
raw_ulpt = pd.read_csv("./data/ulpt.csv", encoding='utf-8')
```

Merge, train, test split data. 

Notice how there is "ULPT" or "LPT" in the title, which is clearly target leakage. To prevent target leakage in the title, I will use regular expressions (Regex) to match permutations of LPT, lpt, ULPT, ulpt and remove.


```python
df = pd.merge(raw_lpt, raw_ulpt, how='outer')
HTML(df.sample(2).to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author</th>
      <th>name</th>
      <th>num_comments</th>
      <th>score</th>
      <th>subreddit</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1002</th>
      <td>socialist111</td>
      <td>t3_abeced</td>
      <td>503</td>
      <td>24466</td>
      <td>UnethicalLifeProTips</td>
      <td>ULPT: Give fake money to homeless people. They will thank you for it, but also when they get arrested and taken to jail, it?ll reduce the number of homeless people in your area.</td>
    </tr>
    <tr>
      <th>1375</th>
      <td>Aukliminu</td>
      <td>t3_ab4d3d</td>
      <td>199</td>
      <td>7180</td>
      <td>UnethicalLifeProTips</td>
      <td>ULPT: Leave bad Amazon reviews on items you buy or any applicable cheap knockoff brands. They might pay you to delete the review.</td>
    </tr>
  </tbody>
</table>




```python
y = df.subreddit
X = df.drop(["subreddit",'name'],axis=1) # drop name, it's a unique identifier, not predictive
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```


```python
def post_to_words(raw_post):
    '''Returns a list of words ready for classification, by tokenizing,
    removing punctuation, setting to lower case and removing stop words.'''
    tokenizer = RegexpTokenizer(r'[a-z]+')
    words = tokenizer.tokenize(raw_post.lower())
    meaningful_words = [w for w in words if not w in set(stopwords.words('english'))]
    return(" ".join(meaningful_words))
```


```python
X_train.loc[:,"title_clean"] = X_train["title"].apply(lambda row : re.sub(r"[uU]*[lL][pP][tT]\s*:*", '', row)).apply(lambda row: post_to_words(row))
```


```python
# for humans
X_test.loc[:,"title_read"] = X_test["title"].apply(lambda row : re.sub(r"[uU]*[lL][pP][tT]\s*:*", '', row))
```


```python
# for model
X_test.loc[:,"title_clean"] = X_test["title_read"].apply(lambda row: post_to_words(row))
```

---
### Modeling

#### CountVectorizer

Let's start simple. CountVectorizer is a bag of words model processes text by ignoring structure of a sentences and merely assesses the count of specific words, or word combinations.


```python
def my_vectorizer(vectorizer,X_train,X_test,y_train,y_test,stop=None):
    '''Takes a vectorizer, fits the model, learns the vocabulary,
    transforms data and returns the transformed matrices'''
    # transform text
    vect = vectorizer(stop_words=stop) 
    train_data_features = vect.fit_transform(X_train.title_clean)
    test_data_features = vect.transform(X_test.title_clean)
    le = LabelEncoder()
    target_train = le.fit_transform(y_train)
    target_test = le.transform(y_test)
    
    # transform non text
    mapper = DataFrameMapper([
    ("author", LabelBinarizer()),
    (["num_comments"], StandardScaler()),
    (["score"], StandardScaler())], df_out=True)
    Z_train = mapper.fit_transform(X_train)
    Z_test = mapper.transform(X_test)
    print(f' Training set learned {train_data_features.shape[1]} distinct vocabulary')
    print(f' Remember: 0 -> {le.classes_[0]}, 1 -> {le.classes_[1]}')
    
    # Baseline model
    print(f' Baseline model that guessed all LPT -> {round(1-sum(target_test)/len(target_test),2)} accurate')
    
    # Combine both df columns together
    a = pd.DataFrame(train_data_features.todense())
    b = Z_train
    c = pd.DataFrame(test_data_features.todense())
    d = Z_test

    # reset indices in order to merge
    a = a.reset_index().drop("index",axis=1)
    b = b.reset_index().drop("index",axis=1)
    c = c.reset_index().drop("index",axis=1)
    d = d.reset_index().drop("index",axis=1)
    
    Z_train = pd.merge(a,b, left_index=True, right_index=True)
    Z_test = pd.merge(c,d, left_index=True, right_index=True)
    return (Z_train, Z_test, target_train, target_test)
```

---
#### Classification

With my data ready in array format, I can now apply binary classifiers. I'll try:

<ul>
<li>Logistic Regression</li>
<li>Naive Bayes Multinomial</li>
<li>Support Vector Machine</li>
</ul>


```python
my_tuple = my_vectorizer(CountVectorizer,X_train,X_test,y_train,y_test,stop='english')
Z_train = my_tuple[0]
Z_test = my_tuple[1]
target_train = my_tuple[2]
target_test = my_tuple[3]
```

     Training set learned 5104 distinct vocabulary
     Remember: 0 -> LifeProTips, 1 -> UnethicalLifeProTips
     Baseline model that guessed all LPT -> 0.5 accurate



```python
def results(model):
    '''Return a sample of 5 wrong predictions'''
    model.fit(Z_train, target_train)
    print(f' Training accuracy: {model.score(Z_train, target_train)}')
    print(f' Test accuracy: {model.score(Z_test, target_test)}')
    predictions = model.predict(Z_test)
    predictions = np.where(predictions==0,"LifeProTips","UnethicalLifeProTips")
    final = pd.DataFrame(list(zip(predictions, y_test, X_test.title_read, X_test.num_comments, X_test.score)), columns=['prediction', 'label', 'title', "num_comments", "score"])   
    wrong = final[final.prediction!=final.label] 
    return HTML(wrong.sample(2).to_html(classes="table table-responsive table-striped table-bordered"))
```


```python
results(LogisticRegression())
```

     Training accuracy: 0.9993021632937893
     Test accuracy: 0.895397489539749





<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>label</th>
      <th>title</th>
      <th>num_comments</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>252</th>
      <td>UnethicalLifeProTips</td>
      <td>LifeProTips</td>
      <td>If you are a parent, record a video message for your children even if you're perfectly healthy.</td>
      <td>824</td>
      <td>11279</td>
    </tr>
    <tr>
      <th>191</th>
      <td>LifeProTips</td>
      <td>UnethicalLifeProTips</td>
      <td>Own two guns. A nice one, registered, and a shitty unregistered handgun</td>
      <td>1169</td>
      <td>7145</td>
    </tr>
  </tbody>
</table>




```python
log_model = LogisticRegression()
log_model.fit(Z_train, target_train);
```

These two were incorrectly classified, because the num_comments might have thrown it off. These are not your typical posts, but by moderators.

Let's look at the largest coefficients that correspond to num_comments (col 6452), score (col 6453), column 576, 4945 & 3125, and peak into the words. 


```python
pd.options.display.float_format = '{:.2f}'.format
my_coef = pd.DataFrame(list(zip(Z_train.columns,abs(log_model.coef_[0]))),columns=["x","coef"]).sort_values(by="coef",ascending=False)
my_coef.head()
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
      <th>x</th>
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6354</th>
      <td>num_comments</td>
      <td>4.33</td>
    </tr>
    <tr>
      <th>6355</th>
      <td>score</td>
      <td>1.83</td>
    </tr>
    <tr>
      <th>576</th>
      <td>576</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>4945</th>
      <td>4945</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>3125</td>
      <td>1.03</td>
    </tr>
  </tbody>
</table>
</div>




```python
my_coef = pd.DataFrame(list(zip(Z_train.columns,log_model.coef_[0])),columns=["x","coef"])
large_coef = my_coef[(my_coef["x"]=="num_comments") | (my_coef["x"]=="score") | (my_coef["x"]==576) | (my_coef["x"]==4945) | (my_coef["x"]==3125)] 
HTML(large_coef.to_html(classes="table table-responsive table-striped table-bordered"))
```




<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>576</th>
      <td>576</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>3125</th>
      <td>3125</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>4945</th>
      <td>4945</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>6354</th>
      <td>num_comments</td>
      <td>-4.33</td>
    </tr>
    <tr>
      <th>6355</th>
      <td>score</td>
      <td>-1.83</td>
    </tr>
  </tbody>
</table>




```python
cvect = CountVectorizer(stop_words='english') 
train_data_features = cvect.fit_transform(X_train.title_clean)
```


```python
print(f' {cvect.get_feature_names()[576]}: {round(np.exp(1.11),2)}x')
print(f' {cvect.get_feature_names()[4945]}: {round(np.exp(1.03),2)}x')
print(f' {cvect.get_feature_names()[3215]}: {round(np.exp(1.05),2)}x')
```

     business: 3.03x
     want: 2.8x
     pitched: 2.86x



```python
f'If the number of comments increases by 1, the likelihood of being an Unethical Life Pro Tip is {round(np.exp(-4.33),2)}x more likely.'
```




    'If the number of comments increases by 1, the likelihood of being an Unethical Life Pro Tip is 0.01x more likely.'




```python
f'If the score (upvotes - downvotes) increases by 1, the likelihood of being an Unethical Life Pro Tip is {round(np.exp(-1.83),2)}x more likely.'
```




    'If the score (upvotes - downvotes) increases by 1, the likelihood of being an Unethical Life Pro Tip is 0.16x more likely.'



This is really performant off the bat with 99.9% training accuracy and 89.5% test accuracy. Yes, there is overfitting but it's picking up signal, in the text.

There's many more false positives than false negatives. One could argue, false positives are not as bad as false negatives since you don't want to heed the advice of a bad tip, but if you miss a life pro tip, it's not as damaging. If we wanted to be more strict, we could tweak the threshold such that only predictions > 75% would be classified as UnethicalLifeProTip.

"Give the same perfume to your wife and your girlfriend. It could save your ass one day." 🙅🏻‍♂️ - _Not a Life Pro Tip_ but was predicted a _Pro Tip_ 

* The more 'popular' i.e. more comments and score, the great likelihood that it is unethical. Controversial posts tend to gain more popularity.
 
* In this training set, if your document includes the word 'business', then the likelihood of being unethical is far more likely by 3x. There's probably a lot of unethical comments around taking advantage of businesses!

---
#### Term Frequency Inverse Document Frequency (TF-IDF)

Compared to CountVectorizer, TF-IDF vectorizer tells us which words are most discriminating between documents.
Words that occur often in one document but don't occur in many documents are important and contain a great deal of discriminating power. Note, TF-IDF figures are between [0,1]. The score is based on how often a word is compared in your document (spam) and other documents.


```python
my_tuple = my_vectorizer(TfidfVectorizer,X_train,X_test,y_train,y_test,stop='english')
Z_train = my_tuple[0]
Z_test = my_tuple[1]
target_train = my_tuple[2]
target_test = my_tuple[3]
```

     Training set learned 5104 distinct vocabulary
     Remember: 0 -> LifeProTips, 1 -> UnethicalLifeProTips
     Baseline model that guessed all LPT -> 0.5 accurate



```python
results(LogisticRegression())
```

     Training accuracy: 0.9567341242149338
     Test accuracy: 0.8535564853556485





<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>label</th>
      <th>title</th>
      <th>num_comments</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>435</th>
      <td>LifeProTips</td>
      <td>UnethicalLifeProTips</td>
      <td>Having an affair? Change your lover's contact name in your phone to "Scam Likely," so your primary partner won't question why you're getting so many calls.</td>
      <td>637</td>
      <td>20447</td>
    </tr>
    <tr>
      <th>241</th>
      <td>UnethicalLifeProTips</td>
      <td>LifeProTips</td>
      <td>Always be gracious when friends or classmates get jobs you both applied to, they might be in a position to hire you in the future</td>
      <td>522</td>
      <td>17866</td>
    </tr>
  </tbody>
</table>



Test accuracy decreases to 85.4% which is lower than CountVectorizer. There are probably not as many discriminating words. Common words are helpful in distinguishing between the two classes. Words of high frequency that are predictive of one of the classes.

It's not entire clear which words are the most influential, some words might indicate sarcasm. Overall, it's impressive that a logistic regression model is so powerful already, let's try a few more algorithms.

---
#### Naive Bayes Classifier
The multinomial Naive Bayes classifier is appropriate for classification with discrete features (e.g., word counts for text classification), as the columns of X are all integer counts.

Note, this classifier accepts only positive values so I have run the abs function on my scaled features. While I have the option to add a prior, I have opted to have Sklearn estimate from training data directly. I don't have a strong opinion if a particular post is in one subreddit over the other. 


```python
model = MultinomialNB()
Z_train = my_tuple[0]
Z_test = my_tuple[1]
target_train = my_tuple[2]
target_test = my_tuple[3]

Z_train.num_comments = abs(Z_train.num_comments)
Z_train.score = abs(Z_train.score)
model.fit(Z_train, target_train)

print(f' Training accuracy: {model.score(Z_train, target_train)}')
print(f' Test accuracy: {model.score(Z_test, target_test)}')

predictions = model.predict(Z_test)
predictions = np.where(predictions==0,"LifeProTips","UnethicalLifeProTips")
final = pd.DataFrame(list(zip(predictions, y_test, X_test.title_read, X_test.num_comments, X_test.score)), columns=['prediction', 'label', 'title', "num_comments", "score"])
wrong = final[final.prediction!=final.label] 
HTML(wrong.sample(2).to_html(classes="table table-responsive table-striped table-bordered"))
```

     Training accuracy: 0.9951151430565248
     Test accuracy: 0.805439330543933





<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>label</th>
      <th>title</th>
      <th>num_comments</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>142</th>
      <td>LifeProTips</td>
      <td>UnethicalLifeProTips</td>
      <td>Server ignoring you hardcore when you?re trying to get your check and pay the bill? Drop a glass/plate onto the floor, the shattering glass will bring them to your table instantly.</td>
      <td>405</td>
      <td>9678</td>
    </tr>
    <tr>
      <th>183</th>
      <td>UnethicalLifeProTips</td>
      <td>LifeProTips</td>
      <td>when making rice, just throw some broccoli on top when there is like 10mins left, it'll be perfectly steamed at the end and is a super easy way to add some nutrition with virtually zero extra work!</td>
      <td>1435</td>
      <td>18783</td>
    </tr>
  </tbody>
</table>




```python
from sklearn.metrics import confusion_matrix
predictions = model.predict(my_tuple[1])
print(confusion_matrix(my_tuple[3], predictions))
tn, fp, fn, tp = confusion_matrix(my_tuple[3], predictions).ravel()
print("True Negatives: %s" % tn)
print("False Positives: %s" % fp)
print("False Negatives: %s" % fn)
print("True Positives: %s" % tp)
```

    [[167  74]
     [ 19 218]]
    True Negatives: 167
    False Positives: 74
    False Negatives: 19
    True Positives: 218


While Naive Bayes also has a high training accuracy, it is severely overfitting. In this case, there are more false negatives than false positives. It tended predict that certain posts were unethical life pro tips, when in fact they are!

---
#### Support Vector Machines

- Exceptional perfomance
- Effective in high-dimensional data
- Low risk of overfitting, but a black box method


```python
my_tuple = my_vectorizer(CountVectorizer,X_train,X_test,y_train,y_test,stop='english')
Z_train = my_tuple[0]
Z_test = my_tuple[1]
target_train = my_tuple[2]
target_test = my_tuple[3]

results(svm.SVC())
```

     Training set learned 5104 distinct vocabulary
     Remember: 0 -> LifeProTips, 1 -> UnethicalLifeProTips
     Baseline model that guessed all LPT -> 0.5 accurate
     Training accuracy: 0.6189811584089323
     Test accuracy: 0.5962343096234309





<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>label</th>
      <th>title</th>
      <th>num_comments</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>337</th>
      <td>UnethicalLifeProTips</td>
      <td>LifeProTips</td>
      <td>When taking apart a vehicle to work on it, take pics of ALL the screws and hardware in its original location. It will prevent ending up with hardware left over when it's all back together.</td>
      <td>779</td>
      <td>17594</td>
    </tr>
    <tr>
      <th>368</th>
      <td>UnethicalLifeProTips</td>
      <td>LifeProTips</td>
      <td>Before checking in at the airport, take a photograph of your luggage. A picture is worth a thousand words if your bags get lost!</td>
      <td>715</td>
      <td>18413</td>
    </tr>
  </tbody>
</table>




```python
params = {'C': [1,3],'gamma': ["scale"]}

grid_search = GridSearchCV(svm.SVC(), param_grid=params, cv=5)
grid_search.fit(Z_train, target_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.score(Z_test,target_test))
```

    0.9204466154919749
    {'C': 3, 'gamma': 'scale'}
    0.9037656903765691



```python
results(svm.SVC(2,gamma="scale"))
```

     Training accuracy: 0.9986043265875785
     Test accuracy: 0.9016736401673641





<table border="1" class="dataframe table table-responsive table-striped table-bordered">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>label</th>
      <th>title</th>
      <th>num_comments</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>435</th>
      <td>LifeProTips</td>
      <td>UnethicalLifeProTips</td>
      <td>Having an affair? Change your lover's contact name in your phone to "Scam Likely," so your primary partner won't question why you're getting so many calls.</td>
      <td>637</td>
      <td>20447</td>
    </tr>
    <tr>
      <th>84</th>
      <td>UnethicalLifeProTips</td>
      <td>LifeProTips</td>
      <td>; Occasionally walk up to your immediate supervisor and ask for some constructive criticism. Be specific by asking something like "What could I have done better on the Penske files?". Whatever they say, just respond thank you and walk away.</td>
      <td>250</td>
      <td>19586</td>
    </tr>
  </tbody>
</table>



Out of the box, SVC is not performant. I have to tune hyperparameters to improve the accuracy. Recall, if C is large, we do not regularize much (larger budget that the margin can be violated), leading to a more perfect classifier of our training data. Of course, there will be a trade off in overfitting and greater error due to higher variance. A smaller gamma helps with lower bias, by trading off with higher variance. Gamma = "scale", which uses `n_features` * `X.var()` tends to work well. 

<table class="table table-striped table-responsive table-bordered">
<thead>
</thead>
<tbody>
    
<tr>
<td><b> Classification Model </b></td>
<td><b> Training Accuracy % </b></td>
<td><b> Test Accuracy % </b></td>
</tr>

<tr>
<td><b>Baseline </b></td>
<td> 0.5 </td>
<td> 0.5 </td>
</tr>   


<tr>
<td><b>Logistic </b></td>
<td> 0.999 </td>
<td> 0.895</td>
</tr>   

<tr>
<td><b>Naive Bayes </b></td>
<td> 0.995 </td>
<td> 0.805 </td>
</tr>  

<tr>
<td><b>Support Vector Machines </b></td>
<td> 0.920 </td>
<td> 0.903 </td>
</tr>   

</tbody>
</table>

Given these results, my selected production model will be the logistic regression model with CountVectorizer as the vectorizer. The Logistic Model is nearly the most performant, but also provides a high level of interpret-ability compared to SVM.

In conclusion:

* The more 'popular' i.e. more comments and score, the great likelihood that it is unethical. Controversial posts tend to gain more popularity.
 
* In this training set, if your document includes the word 'business', then the likelihood of being unethical is far more likely by 3x. There's probably a lot of unethical comments around taking advantage of businesses!

---
### Wrap up

I was able to create an app using Natural Language Processing to classify which subreddit a particular post belongs to.

While this was a fun use case of NLP, this analysis is widely applicable other areas, such as politics in classifying fake news vs. real news, or for eCommerce, with sentiment analysis of user reviews (i.e. polarity classification - positive, negative of neutral). Further, many virtual assistants (Amazon Alexa, Google Assistant) use NLP to understand human questions and provide the appropriate responses.

In these scenarios, there would be far greater consequences, if the prediction was a false-positive or false-negative, so fine tuning the model to adjust for these thresholds is critical.

As a next step, I hope to investigate other NLP open source packages such as [Spacy](https://spacy.io/)!