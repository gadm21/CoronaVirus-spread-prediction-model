import numpy as np
import pandas as pd
from random import randint
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, Activation, TimeDistributed, Dropout, Lambda, RepeatVector, Input, Reshape
from keras.callbacks import ModelCheckpoint 
from keras import backend as K

def normalize_data(data, scaler):
    fitted_scaler= scaler.fit(data) 
    normalized_data= fitted_scaler.transform(data) 
    return normalized_data


def load_data(data, time_step= 2, after_day= 1, train_percent= 0.67):
    seq_length= time_step+ after_day
    result= []

    for index in range(len(data)- seq_length+1):
        result.append(data[index: index+ seq_length])
    
    train_size= int(len(result) * train_percent)
    print("result_shape:", result[0].shape)
    train= result[:train_size, :]
    validate= result[train_size:, :]

    train= train[:, :time_step]
    train_labels= train[:, time_step:]
    validate= validate[:, :time_step]
    validate_labels= validate[:, time_step:]

    return [train, train_labels, validate, validate_labels]


def base_model (feature_len= 3, after_day= 3, input_shape= (8, 1)):
    model= Sequential()
    model.add(LSTM(units= 100, return_sequences= False, input_shape= input_shape))
    model.add(RepeatVector(after_day))
    model.add(LSTM(units= 200, return_sequences= True))
    model.add(TimeDistributed(Dense(units= feature_len, activation= 'linear')))
    return model

def seq2seq (feature_len= 1, after_day= 1, input_shape= (8, 1)):

    '''
    ENCODER
    X : input sequence
    C : LSTM(X), the context vector

    DECODER
    y(t) : LSTM(s(t-1), y(t-1)), where s is the hidden state of LSTM(h and c)
    y(0) : LSTM(s0, C), C is the context vector from the ENCODER
    '''

    #ENCODER
    encoder_inputs= Input(shape= input_shape) #time_steps and features
    encoder= LSTM(units= 100, return_state= True, name= "encoder")
    encoder_outputs, state_h, state_c= encoder(encoder_inputs)
    states= [state_h, state_c]

    #DECODER
    reshaper= Reshape((1, 100), name= "reshaper")
    decoder= LSTM(units= 100, return_sequences= True, return_state= True, name= "decoder")

    #DENSER
    denser_ouput= Dense(units= feature_len, activation= "linear", name= "output")

    inputs= reshaper(encoder_outputs)
    all_outputs= []

    for _ in range(after_day):
        outputs, h, c= decoder(inputs, initial_state= states)

        inputs= outputs
        states= [state_h, state_c] #or maybe states= [h, c]???

        outputs= denser_ouput(outputs)
        all_outputs.append(outputs)
    
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model= Model(inputs= encoder_inputs, outputs= decoder_outputs)

    return model