# Essay Scoring System

This repository contains the implementation of two distinct tasks for essay scoring using natural language processing techniques and machine learning models. The system processes essays to predict scores based on the content's quality and relevance.

## Project Structure

- `main.py`: Main script containing the code for both Task A and Task B.
- `Data/`: Directory containing training and testing datasets.
  - `train.csv`: Training data.
  - `test.csv`: Testing data.
- `Results/`: Directory for storing models' training parameters, images of plots, and prediction scores for the test text.
  - Task-specific subfolders and results files.
- `DeBERTA_initial/`: Initial configuration and weights for the DeBERTa model used in Task B.

## Setup and Installation

Before running the project, ensure that all dependencies are installed. You can install the necessary libraries using:

```
pip install -r requirements.txt
```

## Usage

To run the main script which executes both tasks:

```
python main.py
```

### Task A: TF-IDF Based Scoring

Task A involves preprocessing text data, extracting features using TF-IDF, and then scoring essays with a LightGBM regressor model. Specific steps include:

- Data preprocessing to clean and prepare text data.
- Feature extraction using paragraph, sentence, and word-level analysis.
- Training a LightGBM model to predict scores based on these features.

### Task B: DeBERTa Based Scoring

Task B uses a pre-trained DeBERTa model, fine-tuned on our specific dataset:

- Text data is tokenized using a custom tokenizer.
- The model is trained and evaluated on the data, with the process managed by Hugging Face's `Trainer` API.
- Outputs are scores predicting the quality of essays.

## Results

- Models' performance can be visualized through the generated plots stored in the `Results/` directory.
- Predictions on test data are saved in the same directory, demonstrating the effectiveness of each approach.

## Contributing

Contributions to this project are welcome. Please ensure to update tests as appropriate.

