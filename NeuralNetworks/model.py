import keras
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape, Activation

def get_model():

    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='sigmoid', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='sigmoid', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3,3), activation='sigmoid', padding='same'))
    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))

    print(model.summary())

    return model