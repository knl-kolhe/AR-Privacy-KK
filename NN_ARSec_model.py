from keras.models import Sequential
from keras.layers import Dense

def return_NN_ARSec():
    #Define Neural network
    model = Sequential()
    model.add(Dense(32, input_dim=540, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model