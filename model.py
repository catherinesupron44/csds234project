import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score  
import joblib
import numpy as np
# Load data
print("Loading data...")
df = pd.read_csv('diabetes.csv')

# Split features and target
print("Splitting data...")  
X = df.drop('Diabetes_012', axis=1)  
y = df['Diabetes_012']

# Split train and test set
print("Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Label encode categoricals
print("Label encoding categoricals...")
categoricals = ['HighBP', 'HighChol', 'CholCheck', 'Smoker','Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare','NoDocbcCost', 'DiffWalk', 'Sex']

label_encoder = LabelEncoder()  

for col in categoricals:
    label_encoder.fit(X_train[col])
    X_train[col] = label_encoder.transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])
    
# Train model   
print("Training model...") 
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, random_state=42)
gb.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
print('Accuracy:', accuracy_score(y_test, gb.predict(X_test)))

# Load model
print("Saving model...")
joblib.dump(gb, 'diabetes_model.pkl')


print("Loading saved model...")
gb = joblib.load('diabetes_model.pkl') 

# Sample user input data
#['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
# 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
# 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
user_data = [1, 1, 0, 30, 1, 1, 1, 0, 0, 0, 1, 0, 1, 5, 30, 30, 1, 0, 9, 5, 1]  

print("Making prediction...")
user_df = pd.DataFrame([user_data], columns=X.columns) 
prediction = gb.predict(user_df) 

print('Prediction:', prediction[0])


print(list(X.columns))
print(categoricals)