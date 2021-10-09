# Author: Kristian Rados
# Last modified: August 9th, 2021

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline


"""ADJUSTABLE TRAINING HYPERPARAMETERS"""
seed = 7  # Used to ensure consistent results for algorithms that utilise randomisation

# K-nearest Neighbors
kmin = 1
kmax = 21

# Decision Tree
min_samples_min = 1
min_samples_max = 15

# Naive Bayes
None  # The var_smoothing parameter is still tuned

# Model comparison score formula
comparison_score = lambda accuracy, f1 : (2 * accuracy + 1 * f1) / 3


"""DATA PREPARATION"""
data = pd.read_csv("data2021.student.csv", delimiter=',', header=[0])  # No attributes

# Class obviously should be categorical
data["Class"] = data["Class"].astype('category')

# Convert all attributes with data type 'object', representing non-numeric
# strings, to categorical
converting = []
for att in data:
    if data[att].dtype.name == 'object':
        data[att] = data[att].astype('category')
        converting.append(att)
print("Explicitly converted the following non-numeric attributes to categorical:")
print(converting, '\n')

# Drop the ID attribute, because it doesn't contribute anything to the classification; there is no point to including it
data.drop(['ID'], axis=1, inplace=True)
print("Dropped the ['ID'] attribute\n")

# Remove all attributes with missing values more than the threshold
missing_threshold = 0.8
dropping = []
for att in data:
    missing = data[att].isnull().sum() / data[att].size
    if missing >= missing_threshold and att != 'Class':
        dropping.append(att)
print("Dropping the following attributes (due to too many missing values):")
print(dropping, '\n')
data.drop(dropping, axis=1, inplace=True)

# Convert numeric attributes to categorical when they are likely to be so
unique_threshold = 12
converting = []
for att in data:
    if data[att].dtype.name != 'category':
        if data[att].nunique() <= 10:
            data[att] = data[att].astype('category')
            converting.append(att)
print("Converted the following numeric attributes to categorical (due to few unique values):")
print(converting, '\n')

# Replace all numeric and nominal attributes with missing values less than the threshold with the global mean and mode respectively
impute_threshold = 0.05  # Note that this, as well as the previous step, covers all missing values for this dataset
imputed = []
for att in data:
    missing = data[att].isnull().sum() / data[att].size
    if missing < impute_threshold and missing != 0 and att != 'Class':
        # Replace values with the mode for nominal attributes
        if data[att].dtype.name == 'category':
            data[att].fillna(data[att].mode()[0], inplace=True)
        # Replace values with the mean for numeric attributes
        else:
            data[att].fillna(data[att].mean(), inplace=True)
        imputed.append(att)
print("Imputed mean (for numeric) and mode (for nominal) values for the following attributes (due to missing values):")
print(imputed, '\n')

# Scaling all numeric attributes to the range [0,1] (normalisation)
data_numeric = data.select_dtypes(include='number')
scaled_data_numeric = MinMaxScaler().fit_transform(data_numeric)
data[data_numeric.columns] = scaled_data_numeric
print("Scaled numeric data to range [0,1]\n")

# Remove all attributes with extremely low variance below the threshold
var_threshold = 0.0
dropping = []
for att in data:
    if data[att].dtype.name != 'category':
        var = data[att].var()
        if var <= var_threshold and att != 'Class':
            dropping.append(att)
print("Dropping the following numeric attributes (due to extremely low variance):")
print(dropping, '\n')
data.drop(dropping, axis=1, inplace=True)

# This is done separately to the dropping attributes with low variance as in the case var_threshold > 0.0
dropping = []
for att in data:
    if data[att].dtype.name == 'category':
        if data[att].nunique() == 1:
            dropping.append(att)
print("Dropping the following categorical attributes (due to only having one unique value):")
print(dropping, '\n')
data.drop(dropping, axis=1, inplace=True)

# Detect and delete all duplicate attributes, keeping the first found attribute
data_transposed = data.transpose()
data_transposed_is_duplicate = data_transposed.duplicated(subset=None, keep='first')
duplicate_cols_indices = np.where(data_transposed_is_duplicate == True)[0]
# Convert the tranposed 'row' indices, representing columns, to the relevant attribute names
duplicate_cols = data.columns[duplicate_cols_indices]
print("Dropping the following attributes (due to being duplicates):")
print(duplicate_cols, '\n')
data.drop(duplicate_cols, axis=1, inplace=True)

# Detect and delete all duplicate instances, keeping the first found instance
data_is_duplicate = data.duplicated(subset=None, keep='first')
duplicate_rows = np.where(data_is_duplicate == True)[0]
data.drop(index=duplicate_rows, axis=0, inplace=True)
print("Dropping the following instances (due to being duplicates):")
print([x + 1 for x in duplicate_rows], '\n')  # Correction made for difference between index and original ID attribute

# Display a correlation matrix for numeric attributes
corr_matrix = data.corr()  # Using the 'pearson' method by default
#sn.heatmap(corr_matrix, annot=True)
#plt.show()

# Remove highly correlated numeric attributes automatically
corr_threshold = 0.9
dropping = []
# Retrieve only the upper triangle of the matrix, not including the diagonal line of 1s, as the calc is symmetrical
corr_matrix_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
dropping = [att for att in corr_matrix_upper.columns if any(corr_matrix_upper[att] > corr_threshold)]
print("Dropping the following numeric attributes (due to being highly correlated with another attribute):")
print(dropping, '\n')
data.drop(dropping, axis=1, inplace=True)

y = data.pop('Class')  # Remove the Class attribute from the dataframe
y = y.to_numpy()

# Using one-hot encoding for categorical attributes, as the classification models don't take strings.
# Downside: adds extra 51 'attributes'
# Upside: does not assume that categories are ordinal, something we can't easily determine without
#         domain knowledge
data_encoded = pd.get_dummies(data)
x = data_encoded.to_numpy()

# Split into training and test sets
x_train = x[:-100]
x_test  = x[-100:]
y_train = y[:-100]
y_test  = y[-100:]



"""MODEL SELECTION"""
skf = StratifiedKFold(n_splits=9)  # 9 splits for 100 samples per fold

def show_results(grid_search):
    best_model = grid_search.best_estimator_
    print("Best accuracy:", grid_search.best_score_)
    accuracy = np.mean(grid_search.cv_results_['mean_test_accuracy'])
    f1       = np.mean(grid_search.cv_results_['mean_test_f1'])
    print("Best accuracy average:", accuracy)
    print("Best f1-score average:", f1)
    model_score = {'model' : best_model, 'score' : comparison_score(accuracy, f1)}
    print("Best parameters:", grid_search.best_params_)
    print("Best classifier:", best_model, '\n')
    return model_score


print("=== Training K-nearest Neighbors Classifier... ===")
k_range = list(range(kmin, kmax + 1, 2))
params = {
    'model__n_neighbors' : k_range,
    'model__weights'     : ['uniform', 'distance'],
    'model__metric'      : ['manhattan', 'euclidean', 'chebyshev']
}

#pca = PCA()
smote = SMOTE(random_state=seed, sampling_strategy=1.0)  # Oversampling minority class to match majority class
model = KNeighborsClassifier(n_neighbors=3)
pipeline = Pipeline([('smote', smote), ('model', model)])
#pipeline = Pipeline([('pca', pca), ('smote', smote), ('model', model)])

grid_search = GridSearchCV(pipeline, param_grid=params, cv=skf, return_train_score=True,
                           scoring=['accuracy', 'f1'], refit='accuracy')
grid_search.fit(x_train, y_train)
knn_score = show_results(grid_search)


print("=== Decision Tree Classifier... ===")
min_samples_leaf_range = list(range(min_samples_min, min_samples_max + 1))
params = {
    'model__criterion'        : ['gini', 'entropy'],
    'model__splitter'         : ['best', 'random'],
    'model__min_samples_leaf' : min_samples_leaf_range
}

#pca = PCA()
smote = SMOTE(random_state=seed, sampling_strategy=1.0)  # Oversampling minority class to match majority class
model = DecisionTreeClassifier(random_state=seed)
pipeline = Pipeline([('smote', smote), ('model', model)])
#pipeline = Pipeline([('pca', pca), ('smote', smote), ('model', model)])

grid_search = GridSearchCV(pipeline, param_grid=params, cv=skf, return_train_score=True,
                           scoring=['accuracy', 'f1'], refit='accuracy')
grid_search.fit(x_train, y_train)
decision_tree_score = show_results(grid_search)


print("=== Training Naive Bayes Classifier... ===")
params = {
    'model__var_smoothing' : [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
}

#pca = PCA()
smote = SMOTE(random_state=seed, sampling_strategy=1.0)  # Oversampling minority class to match majority class
model = GaussianNB()
pipeline = Pipeline([('smote', smote), ('model', model)])
#pipeline = Pipeline([('pca', pca), ('smote', smote), ('model', model)])

grid_search = GridSearchCV(pipeline, param_grid=params, cv=skf, return_train_score=True,
                           scoring=['accuracy', 'f1'], refit='accuracy')
grid_search.fit(x_train, y_train)
bayes_score = show_results(grid_search)


print("=== Training Multi-layer Perceptron Classifier... ===")
params = {
    'model__hidden_layer_sizes' : [(100), (100,100), (100,100,100)],
    'model__activation'         : ['tanh', 'relu'],
    'model__solver'             : ['lbfgs', 'sgd', 'adam'],
    'model__alpha'              : [1e-4, 1e-3]
}

#pca = PCA()
smote = SMOTE(random_state=seed, sampling_strategy=1.0)  # Oversampling minority class to match majority class
model = MLPClassifier(random_state=seed, max_iter=3000, learning_rate='constant')
pipeline = Pipeline([('smote', smote), ('model', model)])
#pipeline = Pipeline([('pca', pca), ('smote', smote), ('model', model)])

import time
start_time = time.time()
grid_search = GridSearchCV(pipeline, param_grid=params, cv=skf, return_train_score=True, n_jobs=-1,  # Using all cores
                           scoring=['accuracy', 'f1'], refit='accuracy', verbose=2)
grid_search.fit(x_train, y_train)
mlp_score = show_results(grid_search)
print(f"MLP Training: {time.time() - start_time} seconds")


# Select the two best models for the final prediction
scores = [
    {'name' : "knn",           'result' : knn_score},
    {'name' : "decision_tree", 'result' : decision_tree_score },
    {'name' : "naive_bayes",   'result' : bayes_score},
    {'name' : "mlp",           'result' : mlp_score}
]

score = lambda x : x['result']['score']
scores.sort(reverse=True, key=score)

print("- Model Comparison Scores -")
ii = 1
for x in scores:
    print(str(ii) + '.', str(score(x)) + ':', x['name'])
    ii += 1

model_1 = scores[0]['result']['model']
model_2 = scores[1]['result']['model']



"""FINAL PREDICTION"""
# Train the final models
final_1 = model_1.fit(x_train, y_train)
final_2 = model_2.fit(x_train, y_train)

# Make final predictions
predictions_1 = final_1.predict(x_test)
predictions_2 = final_2.predict(x_test)

# Write to predictions file for submission
with open('predict.csv', 'w') as file:
    file.write("ID,Predict1,Predict2\n")
    ii = 1001
    for n in range(len(predictions_1)):
        file.write(str(ii) + ',' + str(int(predictions_1[n])) + ',' + str(int(predictions_2[n])) + '\n')
        ii += 1

print("\nFinal predictions have been saved to 'predict.csv'")