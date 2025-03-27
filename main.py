import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import pywt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

dataset_path = r"C:\Users\sarat\QR Authentication\Assignment Data-20250326T044007Z-001\Assignment Data"

# Load images
def load_images(folder):
    images = []
    labels = []
    for label in ["First Print", "Second Print"]:
        path = os.path.join(folder, label)
        for file in tqdm(os.listdir(path), desc=f"Loading {label}"):
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)


images, labels = load_images(dataset_path)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

images = images / 255.0

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

X_train_cnn = X_train.reshape(-1, 256, 256, 1)
X_test_cnn = X_test.reshape(-1, 256, 256, 1)

from tensorflow.keras.utils import to_categorical
y_train_cnn = to_categorical(y_train, num_classes=2)
y_test_cnn = to_categorical(y_test, num_classes=2)

# CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()
cnn_history = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=16, validation_data=(X_test_cnn, y_test_cnn))

cnn_model.save('cnn_qr_model.h5')

loss, acc = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print(f"Test Accuracy: {acc:.4f}")