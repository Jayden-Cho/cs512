# WeSTClass with Prompting PLM (RoBERTa-large-MNLI)

## Abstract

This project is about the combination of WeSTClass with zero-shot prompting for document classification. The resulting model, WeSTRo, asssigns weights to the prediction result of WeSTClass and zero-shot classifier based on `RoBERTa-large-MNLI`, either equally or based on the unique evaluation criteria. When tested on news articles and movie reviews, both WeSTRo variants significantly outperformed the standalone WeSTClass model. This study highlights the potential of merging WeSTClass's dynamic data augmentation with RoBERTa's contextual understanding for improved document classification.


## Proposed Idea

1. **Background:** WeSTClass, based on static token embeddings, has demonstrated potential in weakly-supervised settings, producing pseudo-labels for classification.
2. **Integration:** Zero-shot prompting was integrated due to its similarities with WeSTClass's weakly-supervised learning. This method excels in cross-domain scenarios, making it suitable for diverse news categories and movie review sentiment analysis.
3. **Methodology:** The combined approach involves:
   - Equally weighting both models.
   - Assigning distinct weights to each model based on their accuracy on pseudo-document predictions.
   - The combined model is named WeSTRo (WeSTClass-RoBERTa).
  

## Features

- Support for multiple datasets: news and movie dataset.
- Convolutional neural network architectures.
- Pre-training and self-training phases with optional evaluation.
- Utilizes `RoBERTa-large-MNLI` for improved text classification performance.
- Adjustable hyperparameters for learning rate, epochs, and batch size.
- Weighted sum of WeSTClass and PLM predictions for final output.


## Prerequisites

Before running this project, ensure that you have the following requirements installed:

- Python 3.x
- NumPy
- TensorFlow or Keras
- Hugging Face's `transformers` library
- A pre-trained `RoBERTa-large-MNLI model`


## Usage

To run the main script, use the following command:

```bash
python main.py --model [cnn|rnn] --dataset [agnews|yelp|CS512] --pretrain_epochs [number] --update_interval [number] --batch_size [number] --maxiter [number] --delta [float] --gamma [float] --beta [float] --alpha [float] --weighted_sum [True|False] --with_evaluation [True|False] --trained_weights [path to weights] --train [True|False] --test [True|False]
```

Optional flags:

- `--model`: Choose the model architecture (`cnn` or `rnn`).
- `--dataset`: Specify the dataset to use.
- `--pretrain_epochs`: Set the number of epochs for pre-training.
- `--update_interval`: Set the update interval for self-training.
- `...`: Other flags for specific hyperparameters and options.

For instance:

```bash
python main.py --model cnn --dataset yelp --pretrain_epochs 30 --update_interval 50
```


## File Structure Overview

The following represents the key elements of the file structure:

```plaintext
.
├── CS512
│   ├── classes.txt                   # Class labels for the CS512 dataset
│   ├── embedding                     # Directory for embedding files
│   ├── movies.txt                    # Movie data
│   ├── movies_test.txt               # Test data for movies
│   ├── movies_train.txt              # Training data for movies
│   ├── movies_train_labels.txt       # Training labels for movies
│   ├── news.txt                      # News data
│   ├── news_test.txt                 # Test data for news
│   ├── news_train.txt                # Training data for news
│   └── news_train_labels.txt         # Training labels for news
├── LICENSE.txt                       # License information for the project
├── gen.py                            # Script to generate data
├── load_data.py                      # Script to load data for processing
├── main.py                           # Main entry script for running classification
├── model.py                          # Script containing the model definition
├── requirements.txt                  # Required Python packages
├── spherecluster                     # Package for spherical clustering
└── test.sh                           # Shell script for testing the setup
```


## Experimental Results

1. **Datasets:** Experiments used a news article dataset and a movie review dataset.
2. **Preprocessing & Training:** Followed the WeSTClass methodology, with slight modifications for zero-shot prompting integration.
3. **Model Weighting Techniques:** Two approaches were tested:
   - WeSTRo-Non-Weighted: Both models were given equal weights.
   - WeSTRo-Weighted: Weights were assigned based on model accuracy on pseudo-documents.
4. **Performance:** Both WeSTRo models significantly outperformed Vanilla WeSTClass across both datasets. The weighted WeSTRo variant generally showed marginally superior results.

The following table shows the experimental results on WeSTRo-Non-Weighted and WeSTRo-Weighted:

| (acc/f1-micro/f1-macro)      | News dataset  | Movies dataset  |
|-----------------------------|---------------|-----------------|
| Vanilla WeSTClass           | 0.82/0.82/0.8181 | 0.78/0.78/0.7786 |
| WeSTRo-Non-Weighted         | 0.86/0.86/0.8617 | 0.90/0.90/0.8999 |
| WeSTRo-Weighted             | 0.88/0.88/0.8822 | 0.91/0.91/0.8999 |


## Analysis

1. **WeSTClass & Word2Vec:** Despite its static nature, Word2Vec captures substantial semantic information useful for classification, especially under weak supervision.
2. **Synergy of WeSTClass and Zero-shot Prompting:** 
   - WeSTClass excels in dynamic data augmentation via pseudo-labeling.
   - RoBERTa captures intricate contextual relationships.
   - The combined model leverages the strengths of both methodologies for enhanced performance.
   

## Conclusion

The fusion of WeSTClass's pseudo-labeling with RoBERTa's zero-shot prompting delivered promising results across two real-world datasets. The dynamic augmentation provided by WeSTClass, combined with RoBERTa's contextual understanding, establishes a robust classification tool, especially beneficial in weakly-supervised settings.
