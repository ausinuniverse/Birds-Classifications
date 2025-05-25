import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===== PATHS =====
BASE_DIR = "/Users/ausin/Desktop/Datasets/Birds"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
MODEL_PATH = "bird_model.h5"
SUBMISSION_PATH = "submission.csv"

# ===== LOAD DATA =====
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Rename columns for consistency
train_df = train_df.rename(columns={'img_id': 'filename', 'target': 'class'})
test_df = test_df.rename(columns={'img_id': 'filename'})

# Check for missing files
train_df['full_path'] = train_df['filename'].apply(lambda x: os.path.join(TRAIN_DIR, x))
train_df = train_df[train_df['full_path'].apply(os.path.exists)]

print(f"✅ Found {len(train_df)} valid training images")

# ===== DATA GENERATORS =====
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='class',
    target_size=(128, 128),
    class_mode='raw',
    batch_size=32,
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col='filename',
    y_col='class',
    target_size=(128, 128),
    class_mode='raw',
    batch_size=32,
    subset='validation',
    shuffle=True
)

# ===== MODEL DEFINITION =====
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_df['class'].nunique(), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===== TRAINING =====
print("🚀 Starting training...")
model.fit(train_gen, validation_data=val_gen, epochs=10)
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# ===== INFERENCE ON TEST SET =====
print("📦 Loading test data and making predictions...")
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_DIR,
    x_col='filename',
    y_col=None,
    target_size=(128, 128),
    class_mode=None,
    batch_size=32,
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_PATH)
predictions = model.predict(test_gen)
predicted_classes = predictions.argmax(axis=1)

test_df['target'] = predicted_classes
test_df[['filename', 'target']].to_csv(SUBMISSION_PATH, index=False)
print(f"✅ Predictions saved to {SUBMISSION_PATH}")
