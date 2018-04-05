# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import load_model
import flask

import tensorflow as tf

import pandas as pd

from os import listdir, sep
from os.path import abspath, basename, isdir
from sys import argv

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model_type = None           # Where we keep the model type, qualified by hyperparameters used during training
    model = None                # Where we keep the model when it's loaded
    graph = None
    indices = None              # Where we keep the indices of Alphabet when it's loaded

    @classmethod
    def get_indices(cls):
        #Get the indices for Alphabet for this instance, loading it if it's not already loaded
        if cls.indices == None:
            model_type='lstm-gender-classifier'
            index_path = os.path.join(model_path, '{}-indices.npy'.format(model_type))
            if os.path.exists(index_path):
                cls.indices = np.load(index_path).item()
            else:
                print("Character Indices not found.")
        return cls.indices

    @classmethod
    def get_model(cls):
        #Get the model object for this instance, loading it if it's not already loaded
        if cls.model == None:
            model_type='lstm-gender-classifier'
            mod_path = os.path.join(model_path, '{}-model.h5'.format(model_type))
            if os.path.exists(mod_path):
                cls.model = load_model(mod_path)
                cls.model._make_predict_function()
                cls.graph = tf.get_default_graph()
            else:
                print("LSTM Model not found.")
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        mod = cls.get_model()
        ind = cls.get_indices()
        print(type(mod))
        print(type(ind))
        result = []

        if mod == None:
            print("Model not loaded.")
        else:
            if 'max_name_length' not in ind:
                max_name_length = 15
                alphabet_size = 26
            else:
                max_name_length = ind['max_name_length']
                ind.pop('max_name_length', None)
                alphabet_size = len(ind)

            inputs_list = input.strip('\n').split(",")
            num_inputs = len(inputs_list)

            X_test = np.zeros((num_inputs, max_name_length, alphabet_size))

            for i,name in enumerate(inputs_list):
                name = name.lower().strip('\n')
                for t, char in enumerate(name):
                    if char in ind:
                        X_test[i, t,ind[char]] = 1

            with cls.graph.as_default():
                predictions = mod.predict(X_test)

            for i,name in enumerate(inputs_list):
                result.append("M," if predictions[i]>0.5 else "F,")
                print("{} ({})".format(inputs_list[i],"M" if predictions[i]>0.5 else "F"))

        return result

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    #Determine if the container is working and healthy.
    # Declare it healthy if we can load the model successfully.
    health = ScoringService.get_model() is not None and ScoringService.get_indices() is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    #Do an inference on a single batch of data
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        '''
        print(type(data))
        print(data)
        data = StringIO(data)
        print(type(data))
        print(data)
        '''
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.count(",")+1))

    # Do the prediction
    predictions = ScoringService.predict(data)

    result = ""
    for prediction in predictions:
        result = result + prediction

    return flask.Response(response=result, status=200, mimetype='text/csv')
