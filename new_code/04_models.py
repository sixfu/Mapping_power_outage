"""
After labeling our data we can move onto modeling. In this step we will gridsearch a
logistic regression, multinominal naive bayes, and a random forest. We will pickle each of
these models so we can bring them into 06_classify_tweets.py to predict the label for new
tweets pulled in from the bot. These will be used in our multi layered prediction model.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split

# Set random seed
np.random.seed(42)

# Import data
final_csv = pd.read_csv('../data/ready_for_modeling.csv')
print(final_csv.shape)
print(final_csv.columns)

# Train test split
X=final_csv['tweet']
y=final_csv['label']

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=42, stratify = y)

# # Grid Searched Logistic Regression & Pickle
pipe_cvec_lr = Pipeline([
    ('cvec' , CountVectorizer()),
    ('lr' , LogisticRegression(max_iter=10_000))
])

pipe_cvec_params = {
    'cvec__max_features': [2_000, 4_000],
    'cvec__stop_words': [None, 'english'],
    'cvec__ngram_range': [(1,1), (2,2)]
}

gs_cvec_lr = GridSearchCV(pipe_cvec_lr,
                         param_grid=pipe_cvec_params,
                         cv=5)

gs_cvec_lr.fit(X_train,y_train)

# Save model in a pickled file named gridsearch_lr_model.sav
file_name_lr = "gridsearch_lr_model.sav"
pickle.dump(gs_cvec_lr, open(file_name_lr, "wb"))

print(f'log_reg train score {gs_cvec_lr.score(X_train,y_train)}')
print(f'log_reg test score {gs_cvec_lr.score(X_test,y_test)}')

# Grid Searched Naive Bayes & Pickle
pipe_cvec_nb = Pipeline([
    ('cvec' , CountVectorizer()),
    ('nb', MultinomialNB(alpha=.5))
])

gs_cvec_nb = GridSearchCV(pipe_cvec_nb,
                         param_grid=pipe_cvec_params,
                         cv = 5)

gs_cvec_nb.fit(X_train,y_train)

# Save model in a pickled file named gridsearch_nb_model.sav
file_name_nb = "gridsearch_nb_model.sav"
pickle.dump(gs_cvec_nb, open(file_name_nb, "wb"))

print(f'nb train score {gs_cvec_nb.score(X_train,y_train)}')
print(f'nb test score {gs_cvec_nb.score(X_test,y_test)}')

# Grid Searched RandomForest & Pickle
pipe_cvec_rf = Pipeline([
    ('cvec' , CountVectorizer()),
    ('rf' , RandomForestRegressor(n_estimators=200,n_jobs=-1))
])

gs_cvec_rf = GridSearchCV(pipe_cvec_rf,
                         param_grid=pipe_cvec_params,
                         cv=5)

gs_cvec_rf.fit(X_train,y_train)

# Save model in a pickled file named gridsearch_rf_model.sav
file_name_rf = "gridsearch_rf_model.sav"
pickle.dump(gs_cvec_rf, open(file_name_rf, "wb"))

print(f'rf train score {gs_cvec_rf.score(X_train,y_train)}')
print(f'rf test score {gs_cvec_rf.score(X_test,y_test)}')
