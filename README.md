!pip install -q tensorflow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import os
import zipfile
import matplotlib.pyplot as plt

with zipfile.ZipFile('/content/archive.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/trash_data')

base_dir = '/content/trash_data'

# View classes in training folder
print("Classes in Train Folder:", os.listdir(os.path.join(base_dir, 'train')))

# Visualize sample images
def show_samples(class_name, folder='train'):
    path = os.path.join(base_dir, folder, class_name)
    files = os.listdir(path)[:5]
    plt.figure(figsize=(12, 4))
    for i, file in enumerate(files):
        img = tf.keras.preprocessing.image.load_img(os.path.join(path, file), target_size=(224, 224))
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")
    plt.show()

show_samples('plastic')

# Create data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Load EfficientNetV2B2 model
base_model = EfficientNetV2B2(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
base_model.trainable = False

print("EfficientNetV2B2 model loaded successfully.")
