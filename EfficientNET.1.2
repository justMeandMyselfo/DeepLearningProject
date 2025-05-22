import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load pre-trained EfficientNetV2S model
base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom classification layers
x = base_model.output  # Correctly link the base model's output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(101, activation='softmax')(x)  # 101 food categories

# Define the complete model
model = Model(inputs=base_model.input, outputs=x)  # Ensure x is properly connected

# Compile the model with additional metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # Accuracy is included by default

# Load Food-101N dataset (Assuming images are in 'data/' directory)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'Food-101N_release\images', target_size=(224, 224), batch_size=32, subset='training', class_mode='sparse')

val_generator = train_datagen.flow_from_directory(
    'Food-101N_release\images', target_size=(224, 224), batch_size=32, subset='validation', class_mode='sparse')

# Use EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[early_stop])

# Function to compute precision, recall, and F1-score
def evaluate_model(model, data_generator):
    true_labels = []
    pred_labels = []

    for images, labels in data_generator:
        preds = model.predict(images)
        pred_classes = np.argmax(preds, axis=1)

        true_labels.extend(labels)
        pred_labels.extend(pred_classes)

        if len(true_labels) >= data_generator.samples:  # To avoid infinite loops
            break

    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Evaluate the model
evaluate_model(model, val_generator)
