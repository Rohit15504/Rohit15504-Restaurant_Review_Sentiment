# Restaurant Review Sentiment Analysis

This project performs sentiment analysis on restaurant reviews, using Natural Language Processing (NLP) techniques to classify reviews as positive or negative. The analysis is done using a dataset of restaurant reviews taken from Kaggle.

## Dataset

The dataset used for this project is available on [Kaggle](https://www.kaggle.com/datasets/hj5992/restaurantreviews). It contains restaurant reviews labeled as either positive or negative.

## Project Structure

The project is structured as follows:

- `Restaurant_Review_Sentiment.ipynb`: The Jupyter Notebook containing the code for data preprocessing, feature extraction, and model training.
- `dataset/Restaurant_Reviews.tsv`: The dataset file (downloaded from Kaggle) in tab-separated values format.
- `README.md`: This file, providing an overview of the project.

## Installation

To run this project, you'll need to install the following dependencies:

- Python 3.x
- Jupyter Notebook
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `nltk`

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

## Usage

1. **Download the Dataset**

   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hj5992/restaurantreviews) and place it in the `dataset` folder.

2. **Run the Jupyter Notebook**

   Open `Restaurant_Review_Sentiment.ipynb` in Jupyter Notebook and execute the cells to perform sentiment analysis on the restaurant reviews.

3. **Data Preprocessing**

   The code preprocesses the reviews by:
   - Tokenizing the text
   - Removing stop words
   - Lemmatizing the tokens

4. **Feature Extraction**

   It uses the `CountVectorizer` and `TF-IDF` techniques to transform the textual data into numerical features for training.

5. **Model Training**

   A machine learning model is trained using Random Forest Classifier algorithm.
