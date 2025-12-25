import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

img_size = 48
batch_size = 64

train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

train_gen = train_data.flow_from_directory(
    "dataset/train",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = test_data.flow_from_directory(
    "dataset/test",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical"
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_gen, epochs=15, validation_data=test_gen)

model.save("model/emotion_model.h5")
