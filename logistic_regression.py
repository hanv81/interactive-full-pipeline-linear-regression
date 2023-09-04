import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

@st.cache_data
def create_dataset(n_samples):
  x_1 = np.random.rand(n_samples)
  x_2 = np.random.rand(n_samples)
  y = np.array([1 if (i >= 0.4 and j >= 0.4) or i + j >= 1.1 else 0 for i,j in zip(x_1, x_2)])
  X = np.concatenate((x_1.reshape(-1,1), x_2.reshape(-1,1)), axis=1)
  return X, y

e = .1e-10
def bce_loss(y, y_pred):
  return -np.mean(y*np.log(e+y_pred) + (1-y)*np.log(e+1-y_pred))

def accuracy(y, y_pred, threshold=.5):
  y_hat = [0 if i < threshold else 1 for i in y_pred]
  return (y==y_hat).sum()/y.shape[0]

def feed_forward(X, w):
  return 1/(1 + np.exp(-(X*w).sum(axis=1)))

def gradient(X, y, y_pred):
  return (y_pred-y).T @ X

def back_propagation(w, dw, eta):
  return w-eta*dw

def generate_weights(n_features):
  return np.random.rand(n_features+1)

def fit(X, y, ETA, EPOCHS, batch_size=0):
  w = generate_weights(X.shape[1])
  X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
  history = {'loss':[], 'accuracy':[], 'weights':[]}
  for i in range(EPOCHS):
    if batch_size > 0:
      id = np.random.choice(len(y), batch_size)
      XX, yy = X_[id], y[id]
    else:
      XX, yy = X_, y
    y_pred = feed_forward(XX, w)
    loss = bce_loss(yy, y_pred)
    acc = accuracy(yy, y_pred)
    history['weights'].append(w)
    history['loss'].append(loss)
    history['accuracy'].append(acc)
    # if i%50==0:
    #   print(f'iter {i}, loss: {loss}, accuracy: {acc}')
    dw = gradient(XX, yy, y_pred)
    w = back_propagation(w, dw, ETA)

  return history

def draw_result(X, y, history, threshold):
  fig, _ = plt.subplots(1,2)
  fig.set_figheight(2)

  plt.subplot(1,2,1)
  plt.title('Decision Boundary')
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.scatter(X[:, 0], X[:, 1], c=y)

  x1 = np.array([X[:, 0].min()-.05, X[:, 0].max()+.05])
  w = history['weights'][np.argmin(history['loss'])]
  x2 = -(x1*w[0] + w[2])/w[1]
  x2_t = -(x1*w[0] + w[2] + np.log(1/threshold-1))/w[1]
  plt.plot(x1, x2)
  plt.plot(x1, x2_t, linestyle = '--')

  plt.subplot(1,2,2)
  plt.title('History')
  plt.xlabel('Epochs')
  plt.plot(history['loss'], label='Loss')
  plt.plot(history['accuracy'], label='Accuracy')
  plt.legend()
  st.pyplot(fig)

def prepare_data():
  n_samples = st.number_input('Number of Samples', value=200, min_value=100, max_value=10000, step=100)
  x,y = create_dataset(n_samples)
  return x,y

def train(X,y):
  col1, col2, col3 = st.columns(3)
  with col1:
    eta = st.number_input('Learning Rate', max_value=.1, value=.01)
  with col2:
    epochs = st.number_input('Epochs', value=300, step=10, min_value=10)
  with col3:
    batch_train = st.toggle('Mini-Batch GD')
    batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=20, step=5)
  threshold = st.slider('Threshold', min_value=.01, max_value=.99, value=.5, step=.01)
  if not batch_train:
    batch_size = 0

  with st.spinner('Training...'):
    t = time.time()
    history = fit(X, y, eta, epochs, batch_size)
    t = (time.time() - t)*1000
    st.write('Training time:', int(t), 'ms')
    w = history['weights'][np.argmin(history['loss'])]
    X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    y_pred = feed_forward(X_, w)
    loss = bce_loss(y, y_pred)
    acc = accuracy(y, y_pred, threshold)
    st.write('Weights:', *history['weights'][np.argmin(history['loss'])])
    st.write('Loss:', loss)
    st.write('Accuracy:', acc)

  with st.spinner('Visualizing...'):
    draw_result(X, y, history, threshold)

def main():
  st.header('Logistic Regression')
  with st.sidebar:
    X,y = prepare_data()
  
  train(X,y)

if __name__ == "__main__":
  main()