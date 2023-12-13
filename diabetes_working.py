import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import pickle
import pandas as pd
# Load your pre-trained model
model = joblib.load('diabetes_model.pkl')


# Function to predict diabetes
feature_order = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# Map user inputs to order
def get_inputs():
    inputs = {}
    for i, feature in enumerate(feature_order):
        inputs[feature] = entries[i].get()

    return pd.Series(inputs)

# Function to predict diabetes
def predict_diabetes():
    # Get user inputs
    user_inputs = get_inputs()
    user_df = pd.DataFrame([user_inputs]) 

    # Reshape for model 
    user_array = user_df.values.reshape(1, -1)
    
    prediction = model.predict(user_array)

    # Display the result
    result_label.config(text=f"Prediction: {'Diabetes' if prediction[0] == 2 else 'Possible Diabetes' if prediction[0] == 1 else 'No Diabetes' if prediction[0] == 0 else 'Invalid'}")
# Create the main window
root = tk.Tk()
root.title("Diabetes Prediction")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

# Pack the label into the window

# Create input labels and entry widgets
input_labels = [
    'High Blood Pressure (0 = no, 1 = yes)', 'High Cholesterol (0 = no, 1 = yes)', 'Cholesterol Check (0 = no, 1 = yes)',
    'Smoker (0 = no, 1 = yes)', 'Stroke (0 = no, 1 = yes)', 'Heart Disease or Attack (0 = no, 1 = yes)', 'Physical Activity (0 = no, 1 = yes)',
    'Fruit Intake (0 = no, 1 = yes)', 'Vegetable Intake (0 = no, 1 = yes)', 'Heavy Alcohol Consumption (0 = no, 1 = yes)',
    'Any Health Care (0 = no, 1 = yes)', 'No Doctor Due to Cost (0 = no, 1 = yes)', 'General Health (1 is best 5 is worst)',
    'How many days in the past month have you had poor mental health?', 'How many days in the past month have you had poor physical health?', 'Difficulty Walking (0 = no, 1 = yes)',
    'Age (range with 1 = 18-24, 9 = 60-64, 13 = 80+)', 'Sex (0 = female, 1 = male)' , 'BMI', 'Education (1 is grade school - 6 is finished college)', 'Income (scale 1-8 1 < $10000 - 8 > $75,000)'
]

entries = []
for i, label_text in enumerate(input_labels):
   label = ttk.Label(main_frame, text= label_text)  
   label.grid(row=i, column=0)

   entry = ttk.Entry(main_frame)
   entry.grid(row=i, column=1)
   entries.append(entry)

# Create a button to trigger prediction
predict_button = ttk.Button(main_frame, text="Predict Diabetes", command=predict_diabetes)
predict_button.grid(row=len(input_labels), columnspan=2)   

result_label = ttk.Label(main_frame, text="")
result_label.grid(row=len(input_labels) + 1, columnspan=2)

# Run the Tkinter event loop
root.mainloop()

