import os
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import math

app = Flask(__name__)

# Use the correct file name
file_path = os.path.join(os.path.dirname(__file__), 'Weather-for-Boating-Activities.csv')
print("Using file path:", file_path)  # Print file path for verification

try:
    data = pd.read_csv(file_path)
except FileNotFoundError as e:
    print("File not found:", e)
    raise

# Preprocess the data
label_encoders = {}
encoded_data = data.copy()

for column in data.columns:
    le = LabelEncoder()
    encoded_data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split features and target variable
X = encoded_data.iloc[:, :-1]
y = encoded_data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

data_train = data.head(50)

# Fungsi untuk menghitung entropi
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# Fungsi untuk menghitung gain informasi
def info_gain(data, split_attribute_name, target_name="Decision"):
    total_entropy = entropy(data[target_name])
    
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

def save_data_to_csv(data, file_path):
    data.to_csv(file_path,index=False)

@app.route('/')
def index():
    gains = {feature: info_gain(data_train, feature, 'Decision') for feature in data.columns[:-1]}
    data_html = data_train.to_html(classes='data', header="true")
    return render_template('index.html', tables=data_html, gains=gains, accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    
    # Extract input values and transform using the appropriate LabelEncoder for each column
    input_values = []
    for column in X.columns:
        value = input_data[column]
        if value in label_encoders[column].classes_:
            encoded_value = label_encoders[column].transform([value])[0]
        else:
            return render_template('error.html', message=f"Unknown value '{value}' for column '{column}'")
        input_values.append(encoded_value)

    # Predict using the trained classifier
    prediction = clf.predict([input_values])
    decision = label_encoders[y.name].inverse_transform(prediction)[0]  # Convert prediction back to original label
    
    if decision == 'No':
        message = "Keputusan: Tidak berlayar"
    else:
        message = "Keputusan: Berlayar"


    new_data = input_data.copy()
    new_data['Decision'] = decision
    new_data_df = pd.DataFrame([new_data])
    global data
    data = pd.concat([data, new_data_df], ignore_index=True)
    save_data_to_csv(data,file_path)
                       
    return render_template('hasil.html', input_data=input_data, decision=decision, message=message)

if __name__ == '__main__':
    app.run(debug=True)