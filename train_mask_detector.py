from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to Input dataset")
ap.add_argument("-p", "--plot", type = str, default = "plot.png", 
	help = "path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type = str,
	default = "mask_detector.model" ,
	help = "path to output face mask detector model")
args = vars(ap.parse_args())


init_learning_rate = 1e-4
Epochs = 20
Batch_size = 32

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:

	label = imagePath.split(os.path.sep)[-2]

	image = load_img(imagePath, target_size = (224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	data.append(image)
	labels.append(label)


data = np.array(data, dtype = "float32")
labels = np.array(labels)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.20, stratify = labels, 
	random_state = 42)


aug = ImageDataGenerator(
	rotation_range = 20,
	zoom_range = 0.15,
	width_shift_range = 0.2,
	height_shift_range=0.2,
	shear_range = 0.15,
	horizontal_flip = True,
	fill_mode = "nearest")


baseModel = MobileNetV2(weights = "imagenet", include_top = False,
	input_tensor = Input(shape = (224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size =(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = "softmax")(headModel)

model = Model(inputs = baseModel.input, outputs = headModel)


for layer in baseModel.layers:
	layer.trainable = False

print("[INFO] compiling modle...")
opt = Adam(lr=init_learning_rate, decay=init_learning_rate/Epochs)
model.compile(loss="binary_crossentropy", optimizer = opt, metrics =['accuracy'])
# TODO use more metrics

print("[INFO] training training head...")

H = model.fit(
	aug.flow(trainX, trainY, batch_size=Batch_size),
	steps_per_epoch = len(trainX) // Batch_size,
	validation_data = (testX, testY),
	validation_steps = len(testX) // Batch_size,
	epochs = Epochs)


print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=Batch_size)

predIdxs = np.argmax(predIdxs, axis = 1)


print(classification_report(testY.argmax(axis = 1), predIdxs,
	target_names = lb.classes_))

print("[INFO] saving mask detector model")

model.save(args["model"], save_format="h5")

N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label ="train_loss")
plt.plot(np.arange(0, N), H.history['val_loss'], label = 'Val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label = 'train_acc')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label = 'Val_acc')
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])