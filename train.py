from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import glob


def smooth_curve(points, factor=0.8) :
    smoothed_points = []
    for point in points:
        if smoothed_points :
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


base_dir = '/Users/csantoso/AppData/Local/My Private Documents/_BitBucket/DL_python/pack_rec/PackRecognition/rokok_small'
train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')

nb_classes = len(glob.glob(train_dir+"/*"))

print("nb_classes",nb_classes)

# start the model creation

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150, 3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))

# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))


model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))


model.summary()

print("--------------- COMPILE MODEL ---------------")

# Compile
model.compile(loss='categorical_crossentropy', 
                optimizer='rmsprop', 
                metrics=['accuracy'])


print("--------------- DATA AUGMENT ---------------")

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, #Data Augmentation to fight Overfitting
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

print("--------------- Train and Validation GENERATOR ---------------")

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150,150),
    batch_size=20
    # class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150,150),
    batch_size=20
    # class_mode='binary'
)

print("--------------- FIT GENERATOR ---------------")
history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=20
)


model.save('cigarettes1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


print("Plot the Results")

epochs = range(1, len(acc)+1)

plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
plt.title('SMOOTHED Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'ro', label= 'Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'r', label= 'Validation loss')
plt.title('SMOOTHED Training and Validation Loss')
plt.legend()

plt.show()