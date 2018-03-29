

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras import optimizers
from keras.models import Model
from keras import backend as K

import numpy as np
import random
import sys


class VHRED():
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.auto_encoder = self.build_autoencoder(self.encoder, self.decoder)

    def build_autoencoder(self, encoder, decoder):
        return autoencoder

    def train(self):
        pass

    def predict(self):
        pass
