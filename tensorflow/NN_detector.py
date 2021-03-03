import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class FCNet(Model):
    def __init__(self, NR, NT):
        super(FCNet, self).__init__()
        #self.input = tf.keras.Input(shape=(NR,))
        self.fc1 = Dense(2*NT, activation='relu', name='layer1')
        self.fc2 = Dense(2*NT, activation='relu', name='layer2')
        self.fc3 = Dense(2*NT, activation='softmax', name='layer3')

    def call(self, y):
        x1 = self.fc1(y)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        return x3