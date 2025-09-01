import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import SGD

# Set project path and append to sys.path for local imports
project_path = '/content/drive/MyDrive/Training'
if project_path not in sys.path:
    sys.path.append(project_path)

# It's assumed these custom modules are available in your project path
from custom_validate_callback import CustomCallback
from image_datagenerator import DirectoryDataGenerator
from loupe_keras import NetRVLAD
from RoiPoolingConv import RoiPoolingConv
from SelfAttention import SelfAttention
from SeqAttention import SeqSelfAttention
from SpectralNormalizationKeras import ConvSN2D
from se import squeeze_excite_block

# --- Configuration Variables ---
checkpoint_path = "/content/drive/MyDrive/Model/CAP_Xception.1.h5"
batch_size = 8
epochs = 1
image_size = (224, 224)
lstm_units = 128
model_name = "CAP_Xception"
nb_classes = 101
optimizer = SGD(learning_rate=0.0001, momentum=0.99, nesterov=True)
train_dir = "/content/food-101/images/train"
val_dir = "/content/food-101/images/val"
ROIS_resolution = 42
ROIS_grid_size = 3
min_grid_size = 2
pool_size = 7

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Helper Functions ---
def getROIS(resolution=33, gridSize=3, minSize=1):
    coordsList = []
    step = resolution / gridSize
    for column1 in range(0, gridSize + 1):
        for column2 in range(0, gridSize + 1):
            for row1 in range(0, gridSize + 1):
                for row2 in range(0, gridSize + 1):
                    x0 = int(column1 * step)
                    x1 = int(column2 * step)
                    y0 = int(row1 * step)
                    y1 = int(row2 * step)
                    if x1 > x0 and y1 > y0 and ((x1 - x0) >= (step * minSize) or (y1 - y0) >= (step * minSize)):
                        if not (x0 == y0 == 0 and x1 == y1 == resolution):
                            # CORRECTED THIS LINE
                            w = x1 - x0
                            h = y1 - y0
                            coordsList.append([x0, y0, w, h])
    return np.array(coordsList)

def crop(dimension, start, end):
    def func(x):
        if dimension == 1:
            return x[:, start:end]
        return x
    return layers.Lambda(func)

def squeezefunc(x):
    return K.squeeze(x, axis=1)

def stackfunc(x):
    return K.stack(x, axis=1)

# --- Model Definition ---
def create_model():
    base_model = Xception(weights='imagenet', input_tensor=layers.Input(shape=(image_size[0], image_size[1], 3)), include_top=False)
    base_out = base_model.output
    dims = base_out.shape[1:]
    feat_dim = dims[-1] * pool_size * pool_size
    base_channels = dims[-1]

    x = squeeze_excite_block(base_out)
    x_f = ConvSN2D(base_channels // 8, kernel_size=1, padding='same')(x)
    x_g = ConvSN2D(base_channels // 8, kernel_size=1, padding='same')(x)
    x_h = ConvSN2D(base_channels, kernel_size=1, padding='same')(x)
    x_final = SelfAttention(filters=base_channels)([x, x_f, x_g, x_h])

    full_img = layers.Lambda(lambda t: tf.image.resize(t, (ROIS_resolution, ROIS_resolution)), name='Lambda_img_1')(x_final)

    rois_mat = getROIS(resolution=ROIS_resolution, gridSize=ROIS_grid_size, minSize=min_grid_size)
    num_rois = rois_mat.shape[0]

    roi_pool = RoiPoolingConv(pool_size=pool_size, num_rois=num_rois, rois_mat=rois_mat)(full_img)

    jcvs = []
    for j in range(num_rois):
        roi_crop = crop(1, j, j + 1)(roi_pool)
        x_roi = layers.Lambda(squeezefunc, name=f'roi_lambda_{j}')(roi_crop)
        x_roi = layers.Reshape((feat_dim,))(x_roi)
        jcvs.append(x_roi)

    x_final_reshaped = layers.Reshape((feat_dim,))(x_final)
    jcvs.append(x_final_reshaped)

    jcvs_stacked = layers.Lambda(stackfunc, name='lambda_stack')(jcvs)
    x_attention = SeqSelfAttention(units=32, attention_activation='sigmoid', name='Attention')(jcvs_stacked)
    x_reshaped = layers.TimeDistributed(layers.Reshape((pool_size, pool_size, base_channels)))(x_attention)
    x_pooled = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='GAP_time'))(x_reshaped)
    lstm = layers.LSTM(lstm_units, return_sequences=True)(x_pooled)

    y = NetRVLAD(feature_size=128, max_samples=num_rois + 1, cluster_size=32, output_dim=nb_classes)(lstm)
    y = layers.BatchNormalization(name='batch_norm_last')(y)
    y = layers.Activation('softmax', name='final_softmax')(y)

    model = Model(inputs=base_model.input, outputs=y)
    return model

# --- Training Phase ---
model = create_model()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
model.summary()

if os.path.exists(checkpoint_path):
    try:
        model.load_weights(checkpoint_path)
        print(f"Successfully loaded model weights from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load weights: {e}. Training from scratch.")
else:
    print(f"Checkpoint file not found. Training from scratch.")

print("Setting up data generators...")
nb_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
nb_val_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

train_dg = DirectoryDataGenerator(
    base_directories=[train_dir], augmentor=True,
    target_sizes=image_size, preprocessors=preprocess_input,
    batch_size=batch_size, shuffle=True
)
val_dg = DirectoryDataGenerator(
    base_directories=[val_dir], augmentor=False,
    target_sizes=image_size, preprocessors=preprocess_input,
    batch_size=batch_size, shuffle=False
)

print(f"Training images: {nb_train_samples}")
print(f"Validation images: {nb_val_samples}")

print("Setting up callbacks...")
os.makedirs('./Metrics', exist_ok=True)
os.makedirs('/content/drive/MyDrive/Model/', exist_ok=True)

def epoch_decay(epoch, lr):
    if epoch % 50 == 0 and epoch != 0:
        lr = lr / 10
    print(f"EPOCH: {epoch}, LR: {lr}")
    return lr

output_model_path = f'/content/drive/MyDrive/Model/{model_name}.14.h5'
checkpointer = ModelCheckpoint(
    filepath=output_model_path,
    verbose=1,
    save_weights_only=False,
    save_freq='epoch'
)
csv_logger = CSVLogger(f'./Metrics/{model_name}_Training.csv')
validation_freq = 5
custom_val_callback = CustomCallback(val_dg, validation_freq, f'./Metrics/{model_name}')

print("Starting training...")
model.fit(
    train_dg,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[checkpointer, csv_logger, custom_val_callback, LearningRateScheduler(epoch_decay)]
)
