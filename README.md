# Ignacio Deleon Fine-Tuning to a Foundation Model 
# GPT-2 IMDB Sentiment Classification

This project uses the GPT-2 model for sequence classification on the IMDB dataset. The script loads the IMDB dataset, preprocesses it, initializes a GPT-2 model for sequence classification, and performs an initial evaluation.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Datasets

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/gpt2-imdb-sentiment-classification.git
   cd gpt2-imdb-sentiment-classification
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install torch transformers datasets
   ```

## Usage

1. Run the script:
   ```bash
   python3 aimodel.py
   ```

## Script Details

- **Load the dataset**: The script uses the `datasets` library to load the IMDB dataset.
- **Load tokenizer and model**: The GPT-2 tokenizer and model are loaded from the `transformers` library.
- **Set padding token**: If the tokenizer does not have a padding token, it is added.
- **Initialize the model**: The GPT-2 model is initialized for sequence classification, and its embeddings are resized to account for any new tokens.
- **Preprocess the dataset**: The dataset is tokenized with padding and truncation.
- **Define training arguments**: Training arguments are defined for the `Trainer` instance.
- **Create a Trainer instance**: The `Trainer` class is used for training and evaluation.
- **Perform initial evaluation**: The script performs an initial evaluation of the model on the test set and prints the results.

