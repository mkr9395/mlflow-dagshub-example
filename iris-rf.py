import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model
max_depth = 5
n_estimators = 200

# set up a tracking server with dagshub

import dagshub
dagshub.init(repo_owner='mkr9395', repo_name='mlflow-dagshub-example', mlflow=True)


mlflow.set_tracking_uri('https://dagshub.com/mkr9395/mlflow-dagshub-example.mlflow')

# mlflow

mlflow.set_experiment('iris-rf')

with mlflow.start_run():
    
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric('artifact',accuracy)
    mlflow.log_params({'max_depth':max_depth,'n_estimators':n_estimators})
    
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plt.savefig('confusion_matrix.png')
    
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.set_tags({'author':'mohit', 'model':'random forest'})
    
    # log code
    mlflow.log_artifact(__file__)
    
    # log model
    mlflow.sklearn.log_model(rf, 'random_forest_model')
    
    print('accuracy',accuracy)
    
    # logging the dataset in mlflow:
    train_df = pd.DataFrame(X_train)
    test_df = pd.DataFrame(X_test)
    
    train_df.loc[:,'variety'] = y_train 
    test_df.loc[:,'variety'] = y_test 
    
    train_df = mlflow.data.from_pandas(train_df)
    test_df = mlflow.data.from_pandas(test_df)
    
    mlflow.log_input(train_df,"train_data")
    mlflow.log_input(test_df, "validation")
    