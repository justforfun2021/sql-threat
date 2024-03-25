
# Loading Dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import text
from keras.preprocessing import sequence
# import transformers
from fastapi import FastAPI

from tokenizers import BertWordPieceTokenizer

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import xgboost, textblob, string

import numpy as np

from keras.utils import pad_sequences

# Joblib
from joblib import dump
from joblib import load



def load_and_predict(input_features, model_filename = 'PIPE_NBCV.joblib'):
    """
    Loads a machine learning model from a file and uses it to make predictions.

    Parameters:
    - model_filename: The path and name of the file where the model is saved.
    - input_features: The input features to predict on.

    Returns:
    The predictions made by the model.
    """
    # Load the model from the file
    model = load(model_filename)
    
    # Make predictions
    predictions = model.predict([input_features])
    
    return predictions

# # Example usage
# pipe_model_nbcv_test = Pipeline([('transform', count_vect), ('nb', naive_bayes.MultinomialNB())])

# # Example input features for prediction
# input_features = pipe_model_nbcv_test[]  # Adjust this based on your model's expected input

# Load the model and make predictions
predictions = load_and_predict(input_features = ' if 0')

print("Predictions:", predictions)


    





