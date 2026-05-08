
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# 2. Normalize images (0 to 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. Class names
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# 4. Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 5. Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train model
history = model.fit(train_images, train_labels,
                    epochs=5,
                    validation_data=(test_images, test_labels))

# 7. Evaluate model
loss, acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", acc)

# 8. Plot Accuracy Graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'])
plt.show()

# 9. ✈️ Aeroplane Detection
# Find first airplane image (label = 0)
for i in range(len(test_labels)):
    if test_labels[i] == 0:
        img = test_images[i]
        label = test_labels[i]
        break

# Show image
plt.figure(figsize=(4,4))
plt.imshow(img)
plt.title("Test Image (Airplane)")
plt.axis('off')

# Predict
prediction = model.predict(np.expand_dims(img, axis=0))

# Output
print("Actual:", class_names[label[0]])
print("Predicted:", class_names[np.argmax(prediction)])