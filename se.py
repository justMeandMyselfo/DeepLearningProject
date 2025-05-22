from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply

def squeeze_excite_block(input_tensor, ratio=16):
    channel_axis = -1 if K.image_data_format() == "channels_last" else 1
    filters = K.int_shape(input_tensor)[channel_axis]  # Use K.int_shape here

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)

    return multiply([input_tensor, se])
