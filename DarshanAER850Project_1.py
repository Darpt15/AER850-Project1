import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
from joblib import dump, load

# 2.1 - Data Processing
df = pd.read_csv('/Users/unnatipatel/Desktop/TMU Files/AER850 - Intro to Machine Learning/Project_1_Data.csv')
print(df.describe())

# 2.2 - Data Visualization
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Data Behavior Analysis by Step')
# Scatter plot of X vs Y 
axs[0, 0].scatter(df['X'], df['Y'], c=df['Step'], cmap='viridis')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_title('X vs Y (colored by Step)')
# Box plot of Z 
df.boxplot(column='Z', by='Step', ax=axs[0, 1])
axs[0, 1].set_title('Z Distribution by Step')
axs[0, 1].set_ylabel('Z')
# Line plot of X, Y, Z 
axs[1, 0].plot(df.index, df['X'], label='X')
axs[1, 0].plot(df.index, df['Y'], label='Y')
axs[1, 0].plot(df.index, df['Z'], label='Z')
axs[1, 0].set_xlabel('Data Point Index')
axs[1, 0].set_ylabel('Value')
axs[1, 0].set_title('X, Y, Z Values Over Dataset')
axs[1, 0].legend()
# Histogram of Z values
axs[1, 1].hist(df['Z'], bins=30)
axs[1, 1].set_xlabel('Z')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].set_title('Distribution of Z Values')
plt.tight_layout()
plt.show()

#2.3 - Correlation Analysis
# Correlation analysis
correlation_matrix = df.corr(method='pearson')
# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()
# Correlation values with the target variable (Step)
print("Correlation with target variable (Step):")
print(correlation_matrix['Step'].sort_values(ascending=False))

# Group statistics by Step
grouped_stats = df.groupby('Step').agg({
    'X': ['mean', 'std'],
    'Y': ['mean', 'std'],
    'Z': ['mean', 'std', 'min', 'max']
})
print("\nGrouped Statistics by Step:")
print(grouped_stats)

#2.4 - Classification Model Development/Engineering 
X = df[['X', 'Y', 'Z']]
y = df['Step']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# The models and their parameter grids
models = {
    'SVM': (SVC(), {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }),
    'Random Forest': (RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    })
}
# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, f1
# Dictionary to store model performances
model_performances = {}
# Perform Grid Search CV for each model
for name, (model, param_grid) in models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    accuracy, precision, f1 = evaluate_model(grid_search, X_test_scaled, y_test)
    model_performances[name] = {'accuracy': accuracy, 'precision': precision, 'f1': f1}
    print(f"\n{name} - Best parameters: {grid_search.best_params_}")
    print(f"{name} - Performance Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1 Score: {f1:.4f}")
# RandomizedSearchCV for Random Forest
rf = RandomForestClassifier()
param_dist = {
    'n_estimators': np.arange(10, 200),
    'max_depth': [None] + list(np.arange(10, 110, 10)),
    'min_samples_split': np.arange(2, 21),
    'min_samples_leaf': np.arange(1, 11),
    'max_features': ['auto', 'sqrt', 'log2']
}
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=100, cv=5, random_state=42, n_jobs=-1, verbose=1)
random_search.fit(X_train_scaled, y_train)
accuracy, precision, f1 = evaluate_model(random_search, X_test_scaled, y_test)
model_performances['Random Forest (RandomizedSearchCV)'] = {'accuracy': accuracy, 'precision': precision, 'f1': f1}

# 2.5 - Model Performance Analysis
print("\nRandom Forest (RandomizedSearchCV) - Performance Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1 Score: {f1:.4f}")

# Best model based on F1 score
best_model_name = max(model_performances, key=lambda x: model_performances[x]['f1'])
print(f"\nBest performing model based on F1 score: {best_model_name}")

# Confusion matrix for the best model
if best_model_name == 'Random Forest (RandomizedSearchCV)':
    best_model = random_search
else:
    best_model = models[best_model_name][0]
    best_model.fit(X_train_scaled, y_train)  # Ensure the model is fitted

y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Stacking Classifier
svm = SVC(probability=True)
rf = RandomForestClassifier()
stacking_clf = StackingClassifier(
    estimators=[('svm', svm), ('rf', rf)],
    final_estimator=KNeighborsClassifier(),
    cv=5
)

# Train stacking classifier
stacking_clf.fit(X_train_scaled, y_train)

# Evaluate stacking classifier
y_pred_stacking = stacking_clf.predict(X_test_scaled)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
precision_stacking = precision_score(y_test, y_pred_stacking, average='weighted')
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')

print("Stacking Classifier Performance:")
print(f"Accuracy: {accuracy_stacking:.4f}")
print(f"Precision: {precision_stacking:.4f}")
print(f"F1 Score: {f1_stacking:.4f}")

# Confusion matrix for stacking classifier
cm_stacking = confusion_matrix(y_test, y_pred_stacking)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Stacking Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#2.6 Stacked Model Performance Analysis
# Compare with individual models
models = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 2.7 - Model Evaluation
# Load and prepare data
df = pd.read_csv('/Users/unnatipatel/Desktop/TMU Files/AER850 - Intro to Machine Learning/Project_1_Data.csv')
X = df[['X', 'Y', 'Z']]
y = df['Step']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train the best model (assuming Random Forest performed best)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train_scaled, y_train)
# Evaluate the model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Best Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
# Save the model and scaler
dump(best_model, 'best_model.joblib')
dump(scaler, 'scaler.joblib')
# Load the model and scaler
loaded_model = load('best_model.joblib')
loaded_scaler = load('scaler.joblib')
# Predict for the given coordinates
new_data = np.array([[9.375, 3.0625, 1.51],
                     [6.995, 5.125, 0.3875],
                     [0, 3.0625, 1.93],
                     [9.4, 3, 1.8],
                     [9.4, 3, 1.3]])
# Scale the new data
new_data_scaled = loaded_scaler.transform(new_data)
# Make predictions
predictions = loaded_model.predict(new_data_scaled)
print("\nPredictions for the given coordinates:")
for coords, pred in zip(new_data, predictions):
    print(f"Coordinates {coords}: Predicted Step {pred}")

# Load the model and scaler
loaded_model = load('best_model.joblib')
loaded_scaler = load('scaler.joblib')

# New data to predict
new_data = np.array([[9.375, 3.0625, 1.51]])

# Scale the new data
new_data_scaled = loaded_scaler.transform(new_data)

# Make prediction
prediction = loaded_model.predict(new_data_scaled)

print(f"Predicted Step: {prediction[0]}")
