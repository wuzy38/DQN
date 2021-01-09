from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Multiply

def QNetwork(action_size, lr=1e-4):
    frame_input = Input(shape=(84, 84, 4), name='frame_input_layer')
    """ structrue 2 """
    # conv1 = Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", padding="same")(frame_input)
    # conv2 = Conv2D(filters=64, kernel_size=4, strides=2, activation="relu", padding="same")(conv1)
    # conv3 = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", padding="same")(conv2)
    # flatten = Flatten()(conv3)
    # fc = Dense(512, activation="relu")(flatten)
    """ structrue 2 """
    conv1 = Conv2D(filters=16, kernel_size=8, strides=4, activation="relu")(frame_input)
    conv2 = Conv2D(filters=32, kernel_size=4, strides=2, activation="relu")(conv1)
    flatten = Flatten()(conv2)
    fc = Dense(256, activation="relu")(flatten)
    action_qvalue = Dense(action_size, activation="linear", name="output_layer")(fc)
    select_action_input = Input(shape=(action_size,), name='action_input_layer')
    select_action_qvalue = Multiply()([select_action_input, action_qvalue])
    model = keras.models.Model(inputs=[select_action_input, frame_input], outputs=select_action_qvalue)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.Huber(),
    )
    return model

