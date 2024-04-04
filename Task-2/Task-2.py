# Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim

# Simple dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# TensorFlow/Keras model
model_tf = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_tf.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

model_tf.fit(X, y, epochs=1000, verbose=0)

# PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model_pt = Net()

criterion = nn.BCELoss()
optimizer = optim.Adam(model_pt.parameters(), lr=0.01)

X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model_pt(X_torch)
    loss = criterion(output, y_torch)
    loss.backward()
    optimizer.step()

# Evaluating models
# TensorFlow/Keras model prediction
print("\nTensorFlow/Keras Model Predictions:")
print(model_tf.predict(X))

# PyTorch model prediction
print("\nPyTorch Model Predictions:")
with torch.no_grad():
    model_pt.eval()
    print(model_pt(X_torch).numpy())
