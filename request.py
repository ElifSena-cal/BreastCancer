from flask import Flask, request, jsonify
import torch
import numpy as np

import os
import glob
import pandas as pd
from flask_cors import CORS

import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import datetime
import torch
import torch.nn as nn
app = Flask(__name__)
CORS(app)

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Initialize the neural network with the input size
input_size = 30
model = NeuralNetwork(input_size)

# Load the latest saved model from the models directory
def LoadModal():
    model_folder = "models/"
    model_files = glob.glob(os.path.join(model_folder, "*.pth"))
    model_files.sort(key=os.path.getmtime, reverse=True)

    if len(model_files) > 0:
        latest_model_file = model_files[0]     
        loaded_model = torch.load(latest_model_file)
        loaded_model.eval()

        print(f"Successfully loaded the latest saved model '{latest_model_file}'.")
    else:
        print("No saved model found in the folder.")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(data)

        input_data = np.array(data["input_data"], dtype=np.float32)
        LoadModal()
       
        input_tensor = torch.tensor(input_data)

        with torch.no_grad():
            prediction = model(input_tensor).item()
            prediction = float(prediction)

            if prediction >= 0.7:
                predicted_class = "Malignant"  
            else:
                predicted_class = "Benign" 
        
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})

# Class for training the breast cancer detection model
class BreastCancerModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path

    def train_model(self):
        try:
            dataset = pd.read_excel(self.data_path)
            dataset = self.data
            X = dataset.drop(columns=['diagnosis']).values
            y = dataset['diagnosis'].values
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            input_size = X_train.shape[1]
            model = NeuralNetwork(input_size)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_epochs = 100
            for epoch in range(num_epochs):
                outputs = model(torch.FloatTensor(X_train))
                loss = criterion(outputs, torch.FloatTensor(y_train).view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                outputs = model(torch.FloatTensor(X_test))
                predicted = (outputs >= 0.5).float()
                accuracy = (predicted == torch.FloatTensor(y_test).view(-1, 1)).float().mean()

            print(f'Test accuracy: {accuracy.item() * 100:.2f}%')
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            model_folder = "models/"
            model_path = f"model_{timestamp}.pth"
            full_model_path = os.path.join(model_folder, model_path)
            torch.save(model, full_model_path)
            print(f"Model saved on {timestamp}.")
            return jsonify({"message": "true"})
        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)})

# Route for training the model using the uploaded Excel file
@app.route("/train", methods=["POST"])
def train_model_route():
    try:
        data = request.json
        if "file_path" not in data:
            return jsonify({"error": "File path is missing."}), 400
        
        file_path = data["file_path"]
        
        model_trainer = BreastCancerModelTrainer(file_path)
        result = model_trainer.train_model()
        
        return result
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
