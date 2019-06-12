Title: What's for dinner?
Date: 2019-05-31 12:00
Topic: Recommender System
Slug: recipe

Driven by my curiousity of how Netflix, YouTube and Spotify serve personalized recommendations, I decided to learn how to create my own recommender system.

**Machine Learning Problem**: Given a person’s preferences in past recipes, could I predict other new recipes they might enjoy?

I created _Seasonings_, a Recipe Recommender System. The motivation behind this web app is to help users discover personalized and new recipes, and prepare for grocery runs! I received a lot early positive feedback and plan future improvements to the UX and model.

I plan to use this whenever I need a jolt of inspiration in the kitchen!

---

### Demo

All code is available on [GitHub](https://github.com/garrrychan/recipe_recommender_system).

Background & [Presentation Deck](https://github.com/garrrychan/recipe_recommender_system/blob/master/presentation.pdf)

<iframe width="560" height="315" src="https://www.youtube.com/embed/qwFbSFgZMts" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

### Lessons learned
Here are some concepts I discovered while creating my Recipe Recommendation System, and may be useful for anyone building their first recommender system.

<br>

#### 1. Data retrieval is not trivial

While the MovieLens, and Yelp public dataset are both instructive, if you want to create a recommender system, I encourage you to reach for your own data set. 

To appreciate the ins-and-outs of recommender system, you need a large cross section of items and user ratings. I would implore you to gather enough data, so that you can tease out interesting results before validating its performance.

<br>

#### 2. Check the results of your matrix factorization, to ensure it's over a floor baseline

The main hyper-parameter of singular value decomposition (SVD) is the number of components.

```svd = TruncatedSVD(n_components=100)```

A rule of thumb is to use a number of components such that you retain a reasonable amount of ```svd.explained_variance_ratio_```, such as 80%-90%. 

While SVD provides ranked suggestions, some predicted ratings may only be a 2 or a 3. I didn't want to recommend a recipe with a predicted rating of 3, because that's below average compared to Chef John's other recipes. Therefore, I added a rule to ensure that recipes recommended were over a specific threshold >=4.

<br>

#### 3. SVD performs best when the user vector is large

In evaluating my model, SVD performs better when the user vector is large. This is obvious, but cannot be overstated, as my app can only take up to 20 recipe ratings during the onboarding flow. In full recommender systems, ratings are continually updated from subsequent sessions, which is why Netflix _learns_ your preferences the more you use it!

Also, as a corollary, SVD generally performed better for users with less ratings, as it was able to recommend new recipes. Nevertheless, it was useful to train the model, such that the decomposed matrix $M$ could be used on new user matrix $U$.

<br>

#### 4. Balance between prediction, and novelty

Users turn to recommendations for not only recipes that are homogeneous to their preferences, but also for serendipitous recipes they might not immediately think of. In my opinion, it's a strength to make content discovery less algorithmic, and more human-like.

Hybrid recommenders address this problem by solving the shortcomings of individual recommenders.
For example:
<ul>
  <li>Content-based recommenders address the cold start problem</li>
  <li>Collaborative filtering is prone to recommending recipes with less ratings, but I see that as a positive to inject new recipes</li>
  <li>Matrix factorization address curse of dimensionality, when no recipes or users are similar to each other </li>
</ul>

It's no surprised that the algorithm that won Cinematch (Netflix's challenge), used an ensemble of multiple models. Hybrid recommenders can get very complex, with feature weighted linear stacking of the various recommenders, but that was out of scope for my first recommender system.

<br>

#### 5. Rank-based evaluation metrics are preferred

For recommender systems, rank based metrics are preferred over RMSE or accuracy. At the end of the day, it's about recommending new recipes that my user will like, in a ranked fashion, over predicting the exact rating.

I leaned on precision at K and recall at K, and these metrics were higher when the user had more than 3 ratings.

--- 

I had a lot of fun building this, and I gained a greater appreciation of what's going on underneath the hood of all the recommender systems I see on a daily basis. YouTube, please keep recommending me cute cat videos!