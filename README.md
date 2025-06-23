#  Twitter Sentiment Analysis using LSTM and GloVe

This project applies an LSTM-based neural network combined with pre-trained GloVe word embeddings to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments. It's a complete deep learning pipeline for text classification, implemented in Python using TensorFlow and Keras.

---

##  Dataset

- **Source:** [`twitter_training.csv`](./data/twitter_training.csv)
- Preprocessed by:
  - Dropping missing values
  - Filtering for only three sentiment classes: `Positive`, `Negative`, `Neutral`

---

##  Model Architecture

- **Text Preprocessing:**
  - Lowercased input
  - Tokenized using Keras `Tokenizer` (with OOV token support)
  - Padded to a max sequence length of 50
- **Embeddings:**
  - GloVe 6B pre-trained embeddings (100-dimensional)
  - Loaded and used as weights in the Embedding layer
- **Model Layers:**
  - Embedding (with GloVe weights, trainable)
  - LSTM (64 units)
  - Dropout
  - Dense (Softmax for 3-class classification)

---

## Performance

- **Test Accuracy:** ~88%
- EarlyStopping based on validation loss
- Class imbalance handled using `compute_class_weight`

---

##  Sample Predictions

Below are a few examples of how the model performed during evaluation:

Tweet: This is the worst service I've had in years.
True: Negative | Predicted: Negative 

Tweet: Thanks so much for your help! You're amazing!
True: Positive | Predicted: Positive 

Tweet: Wtf?
True: Neutral | Predicted: Negative 

---
##  GloVe Embeddings

This project uses pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings to represent words as dense vectors.

- Specifically, it uses: **`glove.6B.100d.txt`** from the [GloVe 6B](https://nlp.stanford.edu/data/glove.6B.zip) set.

### How to Download

1. Download from: [https://nlp.stanford.edu/data/glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)
2. Unzip the file.
3. Move the `glove.6B.100d.txt` file to the `glove/` folder in the root directory.

## Project Structure
- data/-twitter_training.csv - Raw Twitter sentiment dataset
- models/-sentiment_model.h5 - Trained LSTM model saved in HDF5 format
- SentimentAnalysis_LSTM_GloVe.ipynb - Jupyter notebook containing model training, evaluation, and analysis
- requirements.txt - Python dependencies and package versions
- README.md - Project overview, setup instructions, and usage guide


##  OOV (Out-of-Vocabulary) Analysis

To evaluate GloVe coverage:

- **Training OOV Rate:** ~4.93%
- **Validation OOV Rate:** ~6.30%

Low OOV rates indicate good word coverage from GloVe embeddings.

---

## Author
Twinkle Gupta


