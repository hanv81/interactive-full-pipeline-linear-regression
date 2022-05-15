import glob
import streamlit as st
from zipfile import ZipFile
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.datasets import mnist

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    st.header('Mnist+ Classifier')
    uploaded_file = st.sidebar.file_uploader("Upload data file", type='zip')
    if uploaded_file is not None:
        with ZipFile(uploaded_file, 'r') as zip:
            zip.extractall('data/mnist')
    files = glob.glob('data/mnist/*.png')
    lst = []
    for f in files:
        im_frame = ImageOps.grayscale(Image.open(f))
        np_frame = np.array(im_frame.getdata())
        np_frame = np_frame.reshape(28,28)
        np_frame = np_frame.astype('uint8')
        lst += [np_frame]
    
    lst = np.array(lst)
    x_train = np.concatenate((x_train, lst), axis=0)
    print(x_train.shape)

if __name__ == "__main__":
    main()