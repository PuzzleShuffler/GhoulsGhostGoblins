##### Load Modules #####
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder,
                                   PolynomialFeatures, StandardScaler)

##### Load dataset #####
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##### Pipeline Setup #####
X_test = test.drop(['id', 'color'], axis='columns')

# split into train, test data
X_train, X_val, y_train, y_val = train_test_split(
    train.drop(['id', 'type', 'color'], axis='columns'),
    train['type'],
    stratify=train['type'],
    test_size=0.30,
    random_state=2022
    )

# Separate Columns by type
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

# Numeric Feature Pipeline
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('scaler', StandardScaler()))
numeric_pipeline_steps.append(('poly', PolynomialFeatures(degree=2)))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# Categorical Feature Pipeline
categorical_pipeline_steps = []
categorical_pipeline_steps.append(('onehot', OneHotEncoder(min_frequency=1, handle_unknown='infrequent_if_exist')))
categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

# Preprocessing Transformer
transformer_steps = []
transformer_steps.append(('cat', categorical_pipeline, categorical_features))
transformer_steps.append(('num', numeric_pipeline, numeric_features))
preprocessing_transformer=ColumnTransformer(transformers=transformer_steps)

# Display Transformer
set_config(display='diagram')
preprocessing_transformer

##### Model Prep #####
# transform data
transformed_X_train = preprocessing_transformer.fit_transform(X_train)
transformed_X_val = preprocessing_transformer.transform(X_val)
transformed_X_test = preprocessing_transformer.transform(X_test)

# label encode target

le = LabelEncoder()
transformed_y_train = le.fit_transform(y_train)
transformed_y_val = le.fit_transform(y_val)

transformed_y_train = np_utils.to_categorical(transformed_y_train)
transformed_y_val = np_utils.to_categorical(transformed_y_val)

##### Define Model #####
def keras_model():
    # create model
    model = Sequential()
    model.add(Dense(21, input_dim=transformed_X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(18, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(12, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(9, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.30))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Early Stop Parameters & Model Checkpoint
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('models/best_mlp_model.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)

training_model = keras_model()

##### Model #####
# Run Training Model
training_model_history = training_model.fit(
    transformed_X_train, transformed_y_train,
    batch_size=50,
    validation_data=(transformed_X_val, transformed_y_val),
    epochs=1_000,
    callbacks=[es, mc])

# summarize history for Accuracy
sns.lineplot(training_model_history.history['accuracy'], marker='o', color='black', label='Train')
sns.lineplot(training_model_history.history['val_accuracy'], marker='o', color='orange', label='Val')
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.show()

# summarize history for Loss
sns.lineplot(training_model_history.history['loss'], marker='o', color='black', label='Train')
sns.lineplot(training_model_history.history['val_loss'], marker='o', color='orange', label='Val')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#### Loading Best Model #####
# Load Weights
best_model = keras_model()
best_model.load_weights('models/best_mlp_model.h5')

scores = best_model.evaluate(transformed_X_val, transformed_y_val)
scores

##### Predictions ons Validation #####
preds_prob = best_model.predict(transformed_X_val)
coded_pred_classes = np.argmax(preds_prob, axis=1)
preds = le.inverse_transform(coded_pred_classes)

ConfusionMatrixDisplay.from_predictions(
    y_true=y_val,
    y_pred=preds,
    cmap=plt.cm.Blues,
    colorbar=False,
)
plt.suptitle(f'Confusion Matrix')
plt.title(f'w/Accuracy: {round(scores[1],3)}')
plt.show()