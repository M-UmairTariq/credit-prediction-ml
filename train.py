import pandas as pd
import mlflow
from sklearn.metrics import precision_score, recall_score ,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("D:/Software Engineering/BAVE INTERNSHIP/Code/Day 7/Credit-Score/bank.csv" , sep = ";")


#experiment name
mlflow.set_experiment("AI - Credit Score")

#separting our X and Y - independent and dependent
X = df.drop("y" , axis= 1)
y = df["y"]

#Separate dataset into train and test
#random state is to keep the dataset same when we run the code again , test_size is to split the dataset into train and test 30% is testdata and 70% is train data
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 42)


#to select categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns

#to select numerical columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

print(categorical_cols)
print(numerical_cols)

with mlflow.start_run():

    #preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", "passthrough", numerical_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    #defined model RandomForestClassifier
    model = RandomForestClassifier(random_state=42)

    #create pipeline 
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    #train model
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test , y_pred)}")
    precision = precision_score(y_test , y_pred , pos_label='yes')

    recall = recall_score(y_test , y_pred , pos_label='yes')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    mlflow.log_metric("Accuracy" , accuracy_score(y_test , y_pred))
    mlflow.log_metric("Precision" , str(precision))
    mlflow.log_metric("Recall" , str(recall))

    mlflow.sklearn.log_model(pipeline, "pipeline_random_forest")


    #TODO: make inferences from mlflow model file , and make fastapi server for it
    