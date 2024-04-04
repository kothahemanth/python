import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing

# Load and preprocess IMDB movie review dataset
max_features = 10000
maxlen = 200
embedding_dim = 128

(train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=max_features)
train_data = preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = preprocessing.sequence.pad_sequences(test_data, maxlen=maxlen)

# Define RNN model with LSTM layers
model = models.Sequential([
    layers.Embedding(max_features, embedding_dim, input_length=maxlen),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer with dropout
    layers.Dense(1, activation='sigmoid')
])

# Define RNN model with GRU layers
# model = models.Sequential([
#     layers.Embedding(max_features, embedding_dim, input_length=maxlen),
#     layers.GRU(64, dropout=0.2, recurrent_dropout=0.2),  # GRU layer with dropout
#     layers.Dense(1, activation='sigmoid')
# ])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print("\nTest accuracy:", test_acc)
