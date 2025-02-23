##### Import Libraries #####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (RandomizedSearchCV,
                                     StratifiedShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder,
                                   PolynomialFeatures)
from xgboost import XGBClassifier

##### Load Data #####
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

##### Data Cleaning #####
# Drop id
train.drop(['id'], axis='columns', inplace=True)
test.drop(['id'], axis='columns', inplace=True)

# Drop color
train.drop(['color'], axis='columns', inplace=True)
test.drop(['color'], axis='columns', inplace=True)

##### Split-out Validation Dataset #####
# Split into X, y
X = train.drop(['type'], axis='columns')
y = train['type']

# Label Encode y
le = LabelEncoder()
y = le.fit_transform(y)

# Train & Validation Sets
X_train, X_validate, y_train, y_validate = train_test_split(X, y, stratify=y,test_size=0.30)

##### Model Set Ups #####
# Separate Columns by type
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Numeric Feature Pipeline (Poly)
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('poly', PolynomialFeatures(degree=4)))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# Preprocessing Transformer
preprocessing_transformer_steps = []
preprocessing_transformer_steps.append(('poly', numeric_pipeline, numeric_features))
preprocessing_transformer=ColumnTransformer(transformers=preprocessing_transformer_steps)

# Display Transformer
set_config(display='diagram')
preprocessing_transformer

##### Spot-Check Algorithms #####
# Store Results
results = []
names = []
scoring = 'accuracy'

# Models
models = []
models.append(('lr', LogisticRegression(solver='liblinear')))
models.append(('gbr', GradientBoostingClassifier()))
models.append(('lgbm', LGBMClassifier()))
models.append(('xgb', XGBClassifier(objective='multi:softprob', eval_metric=('mlogloss'), use_label_encoder=False)))
models.append(('rf', RandomForestClassifier()))

for name, model in models:
    # CV
    cv = StratifiedShuffleSplit(n_splits=10)
    # Pipeline
    model_pipeline_steps = []
    model_pipeline_steps.append(('transformer', preprocessing_transformer))
    model_pipeline_steps.append(('model', model))
    model_pipeline = Pipeline(steps=model_pipeline_steps)
    # CV results
    cv_results = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring=scoring)
    # Append Results
    results.append(cv_results)
    names.append(name)
    # Results Mean +/- std
    msg = "%s: %f +/- %f" % (name, cv_results.mean(), cv_results.std())
    # Print Results
    print(msg)
    
# Algorithm Comparison Boxplot
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

##### Tune Best Model, Fit on Validation Set #####
# Model 
model = LogisticRegression(solver='liblinear', max_iter=1000)

# Scoring
scoring = 'accuracy'

# Pipelinee Steps
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', model))
model_pipeline = Pipeline(steps=model_pipeline_steps)

# Param Grid
param_grid = {'model__C': np.linspace(0.001, 10, 10000),
              'model__penalty':['l1', 'l2']}

# RandomizedSearch CV
rscv_model = RandomizedSearchCV(estimator=model_pipeline, param_distributions=param_grid, scoring=scoring, n_iter=100)

# Display Model
rscv_model

# Fit Model on Train
rscv_model.fit(X_train, y_train)

# Confusion Matrix
plot_confusion_matrix(rscv_model,
                      X_validate,
                      y_validate,
                      cmap=plt.cm.Blues,
                      display_labels=list(le.classes_),
                      colorbar=False)
plt.show()

# Best
print('best params:', str(rscv_model.best_params_))
print('best score: %f' % rscv_model.best_score_)

# Accuracy on Validation Set
print('Accuracy on Validation Set: %f' % accuracy_score(y_validate, rscv_model.predict(X_validate)))

##### Standalone Model on Entire Training Set #####
# Model
model = LogisticRegression(solver='liblinear', max_iter=1000)

# Model Pipeline
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', model))
final_model = Pipeline(steps=model_pipeline_steps).set_params(**rscv_model.best_params_)

# Display Model
final_model

# Fit Model
final_model.fit(X,y)

# Predictions
predictions = le.inverse_transform(final_model.predict(test))

# Submission
submission = pd.DataFrame({'id':sample_submission['id'], 'type':predictions})
submission.to_csv('submissions/lr v3 2021-05-16.csv', index=False)

##### Create Stacking Model on Entire Training Set #####
# Models to Stack
gbr = GradientBoostingClassifier(n_estimators=10000,
                                 learning_rate=0.01)

lgbm = LGBMClassifier(n_estimators=10000, 
                      learning_rate=0.01)

rf = RandomForestClassifier(n_estimators=10000)

lr = LogisticRegression(solver='liblinear', max_iter=1000)

xgb = XGBClassifier(n_estimators=10000,
                    learning_rate=0.01,
                    objective='multi:softprob', 
                    eval_metric=('mlogloss'),
                    use_label_encoder=False)

# Stacking Model
model = StackingCVClassifier(classifiers=(lr, rf, gbr, lgbm, xgb), 
                            meta_classifier=xgb, 
                            use_features_in_secondary=True,
                            n_jobs=-1)

# Model Pipeline
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', model))
stacking_model = Pipeline(steps=model_pipeline_steps)

# Display Model
stacking_model

# Fit Model
stacking_model.fit(X,y)

# Predictions
predictions = le.inverse_transform(stacking_model.predict(test))

# Submission
submission = pd.DataFrame({'id':sample_submission['id'], 'type':predictions})
submission.to_csv('submissions/stack v2 2021-05-16.csv', index=False)
