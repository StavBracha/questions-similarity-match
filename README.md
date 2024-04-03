## README for Question Pair Similarity Model

## Description

The Question Pair Similarity Model is designed to assess the similarity between two questions by computing their semantic similarity. This is particularly useful in applications such as question-answering systems, chatbots, and knowledge base management, where identifying duplicate or semantically similar questions can significantly improve information retrieval and user experience.

The model employs pre-trained word embeddings, custom Dataset handling, a novel Co-Attention mechanism, and a contrastive loss function to train a neural network that can predict the similarity between question pairs.

## Installation

Prerequisites
Python 3.6 or newer
PyTorch 1.7.1 or newer
TensorFlow 2.4 or newer
Pandas
NumPy
Scikit-learn

## To use the Question Pair Similarity Model, follow these steps:

1) Prepare Your Data: Your dataset should be in a CSV format with three columns: question1, question2, and is_duplicate, where is_duplicate indicates whether the questions are considered semantically similar.
2) Pre-trained Word Embeddings: Place your pre-trained word embeddings file in the project directory and specify its name. The embeddings should be in a format where each line contains a word followed by its vector representation, separated by spaces.
3) Train the Model:
* Modify the dataset path in the script to point to your dataset file.
* Load pre-trained embeddings by calling load_pre_trained_embeddings(path_to_embeddings).
* Initialize the Dataset class instances for training, validation, and testing datasets.
*Configure and instantiate the model, specifying the required parameters such as the dimensions of the embeddings, the number of heads for the Co-Attention mechanism, and the depth of the encoder layers.
* Train the model using the provided training loop, adjusting parameters like learning rate and batch size as needed.
* Evaluate the Model: Use the evaluation loop to assess the performance of the model on a validation or test set, obtaining metrics such as loss and cosine similarity for true and false pairs.


The script is designed to be flexible, allowing for adjustments and customizations. You can modify the architecture, switch out components, adjust hyperparameters, or use different pre-trained embeddings based on your specific requirements or datasets.
