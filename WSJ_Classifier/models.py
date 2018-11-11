# -*- coding: utf-8 -*-
"""Deep Learning Models for news article summary classification

This file contains the definition of three slightly different models
for news article summary classification.

"""

from keras.layers import Input, Dense, GRU, TimeDistributed
from keras.layers import Bidirectional, Dropout, LSTM
from keras.models import Model

def model_1(seq_length, embedding_layer, num_categories):
    """Model with LSTM cells but without dropouts
    
    Args:
        seq_length (int): Number of words per input sequence
        embedding_layer (keras.layers.Embedding): Word embedding layer
        num_categories (int): Number of news article summary
                              categories (sections)
                              
    Returns:
        model (keras.models.Model): Keras neural network
    
    """
    model_input = Input(shape=(seq_length,), dtype='int32')
    embed = embedding_layer(model_input)
    fc = TimeDistributed(Dense(100, activation="relu"))(embed)
    rnn = Bidirectional(LSTM(units=100, return_sequences=False))(fc)
    den = Dense(512, activation="relu")(rnn)
    out = Dense(num_categories, activation="softmax")(den)

    model = Model(model_input, out)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    return model

def model_2(seq_length, embedding_layer, num_categories):
    """Model with LSTM cells and dropout layers
    
    Args:
        seq_length (int): Number of words per input sequence
        embedding_layer (keras.layers.Embedding): Word embedding layer
        num_categories (int): Number of news article summary
                              categories (sections)
                              
    Returns:
        model (keras.models.Model): Keras neural network
    
    """
    model_input = Input(shape=(seq_length,), dtype='int32')
    embed = embedding_layer(model_input)
    fc = TimeDistributed(Dense(100, activation="relu"))(embed)
    rnn = Bidirectional(LSTM(units=100, return_sequences=False))(fc)
    drop_1 = Dropout(0.5)(rnn)
    den = Dense(512, activation="relu")(drop_1)
    drop_2= Dropout(0.5)(den)
    out = Dense(num_categories, activation="softmax")(drop_2)

    model = Model(model_input, out)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    return model

def model_3(seq_length, embedding_layer, num_categories):
    """Model with GRU cells and dropout layers
    
    Args:
        seq_length (int): Number of words per input sequence
        embedding_layer (keras.layers.Embedding): Word embedding layer
        num_categories (int): Number of news article summary
                              categories (sections)
                              
    Returns:
        model (keras.models.Model): Keras neural network
    
    """
    model_input = Input(shape=(seq_length,), dtype='int32')
    embed = embedding_layer(model_input)
    fc = TimeDistributed(Dense(100, activation="relu"))(embed)
    rnn = Bidirectional(GRU(units=100, return_sequences=False))(fc)
    drop_1 = Dropout(0.5)(rnn)
    den = Dense(512, activation="relu")(drop_1)
    drop_2= Dropout(0.5)(den)
    out = Dense(num_categories, activation="softmax")(drop_2)

    model = Model(model_input, out)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
    return model
