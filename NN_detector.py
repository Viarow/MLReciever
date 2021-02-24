import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class FCNet(Model):
    def __init__(self, NR, NT):
        super(FCNet, self).__init__()
        self.input = tf.keras.Input(shape=(NR,))
        self.fc1 = Dense(NT, activation='relu')
        self.fc2 = Dense(NT, activation='softmax')

    def call(self, y):
        x = self.fc1(y)
        x = self.fc2(x)
        return x