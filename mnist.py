import glob
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile
from PIL import Image, ImageOps
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    st.header('Mnist+ Classifier')
    uploaded_file = st.sidebar.file_uploader("Upload data file", type='zip')
    if uploaded_file is not None:
        with ZipFile(uploaded_file, 'r') as zip:
            zip.extractall('data/mnist')

        files = glob.glob('data/mnist/*.png')
        x_new = []
        y_new = [10 for _ in range(len(files))]
        for f in files:
            im_frame = ImageOps.grayscale(Image.open(f))
            im_frame = ImageOps.invert(im_frame)
            np_frame = np.array(im_frame.getdata())
            np_frame = np_frame.reshape(28,28)
            x_new += [np_frame]

        x_new = np.array(x_new).astype('uint8')
        y_new = np.array(y_new).astype('uint8')
        x_train = np.concatenate((x_train, x_new), axis=0)
        y_train = np.concatenate((y_train, y_new), axis=0)
        print(x_train.shape, y_train.shape)

        rows = 11
        cols = 5
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(7,15))
        for row in range(rows):
            for col in range(cols):
                random_index = np.random.choice(np.where(y_train == row)[0])
                axs[row][col].grid('off')
                axs[row][col].axis('off')
                axs[row][col].imshow(x_train[random_index], cmap='gray')
        st.write(fig)

        epochs = st.text_input('Epochs', 10)
        cnn = st.checkbox('CNN')
        if st.button("Train"):
            x_train = x_train / 255.0
            x_test = x_test / 255.0
            y_train_encode = to_categorical(y_train, num_classes=11)
            y_test_encode = to_categorical(y_test, num_classes=11)

            model = Sequential()
            if cnn:
                x_train = x_train[..., None]
                x_test = x_test[..., None]
                model.add(Conv2D(8, 3, padding='same', activation='relu', input_shape=x_train.shape[1:]))
                model.add(Conv2D(8, 3, padding='same', activation='relu'))
                model.add(MaxPooling2D())
                model.add(Flatten())
            else:
                model.add(Flatten(input_shape=x_train.shape[1:]))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(11, activation='softmax'))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
            history = model.fit(x_train, y_train_encode, epochs=int(epochs), verbose=1)
            model.evaluate(x_train, y_train_encode)
            model.evaluate(x_test, y_test_encode)

if __name__ == "__main__":
    main()