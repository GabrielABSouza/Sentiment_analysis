🎬 IMDB Sentiment Analysis with LSTM (Keras)
This project is a Sentiment Analysis classifier built using Keras and TensorFlow. It uses the IMDB movie review dataset to train an LSTM (Long Short-Term Memory) neural network to classify reviews as positive or negative.

📌 Overview

This project demonstrates:

Loading and preprocessing the IMDB dataset

Building a deep learning model using Embedding + LSTM layers

Training the model on text data

Evaluating the model on unseen data

Predicting sentiment for a sample of test reviews


🧠 Model Architecture

The model consists of:

An Embedding layer to learn word embeddings from the input data

Two stacked LSTM layers to capture temporal dependencies in the review text

A final Dense layer with a sigmoid activation for binary classification


🗂 Dataset

IMDB Movie Reviews (50,000 reviews: 25,000 train / 25,000 test)

Pre-tokenized and preprocessed using keras.datasets.imdb

Only the top 10,000 most frequent words are used

Reviews are padded to 100 words for uniform input size


📈 Training Performance

After training for 3 epochs:

Training Accuracy: ~93%

Validation Accuracy: ~84%


🧪 Sample Predictions

The model is tested on a subset of the test data:

Actual Sentiments:    [1 0 0 0 1 0 1 0 0 0 1]
Predicted Sentiments: [1 0 0 0 1 0 1 0 0 0 1]
✅ Prediction matches all 11 out of 11 samples.

🚀 How to Run

1. Clone this repository:
git clone https://github.com/yourusername/imdb-sentiment-lstm.git
cd imdb-sentiment-lstm

2. Install dependencies

pip install tensorflow numpy keras

3. Run the script

python sentiment_analysis.py


📦 Dependencies
tensorflow

keras

numpy

📌 Notes
input_length in Embedding is now deprecated in Keras 3.x, but is safe to ignore.

The model can be extended with dropout layers or bidirectional LSTMs for better generalization.
