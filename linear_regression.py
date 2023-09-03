import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples, w, b):
  x = np.random.randn(n_samples,1)
  noise = np.random.randn(n_samples,1)
  y = w*x + noise + b
  return x, y

def visualize_loss_surface(x, y, w0, b0, history):
  w_ = np.array(history['weights'])
  y_pred = w_[:,0]*x + w_[:,1]
  loss_ = ((y_pred - y)**2).mean(axis=0)
  w = np.linspace(w0-3, w0+3, 200)
  b = np.linspace(b0-3, b0+3, 200)
  ww, bb = np.meshgrid(w, b)
  wb = np.c_[ww.ravel(), bb.ravel()]
  loss = np.mean(((wb[:,0]*x + wb[:,1])-y)**2, axis=0)

  fig = go.Figure(data=[go.Surface(x=w, y=b, z=loss.reshape(ww.shape)),
                        go.Scatter3d(x=w_[:,0], y=w_[:,1], z=loss_, mode='markers')])
  st.plotly_chart(fig)

def prepare_data():
  n_samples = st.slider('Select Data Size', value=2000, min_value=100, max_value=20000, step=100)
  w = st.number_input('w', value=3)
  b = st.number_input('b', value=2)
  x,y = generate_data(n_samples, w, b)
  return x,y,w,b

def mse(y, y_pred):
  return ((y-y_pred)**2).mean()

def feed_forward(x, w):
  return (x@w).reshape(-1,1)

def gradient(x, y, y_pred):
  return 2*(x*(y_pred - y)).mean(axis=0)

def generate_weights(n_features):
  return np.random.rand(n_features+1)

def draw_result(x, y, history):
  w = history['weights'][-1]
  x_line = np.array([x.min(), x.max()])
  y_line = x_line * w[0] + w[1]

  fig, _ = plt.subplots(1,2)
  fig.set_figheight(2)
  plt.subplot(1,2,1)
  plt.title('Regression Line')
  plt.scatter(x,y)
  plt.plot(x_line, y_line, c='r')

  plt.subplot(1,2,2)
  plt.title('Loss')
  plt.plot(history['loss'])
  st.pyplot(fig)

def fit(x, y, eta, epochs, batch_size=0):
  w = generate_weights(x.shape[1])
  history = {'loss':[], 'weights':[]}
  x_ = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
  for i in range(epochs):
    y_pred = feed_forward(x_, w)
    loss = mse(y, y_pred)
    history['loss'].append(loss)
    history['weights'].append(w)
    if i%10==0:
      print(f'iter {i}, loss: {loss}')
    if batch_size > 0:
      id = np.random.choice(len(y), batch_size)
      dw = gradient(x_[id], y[id], y_pred[id])
    else:
      dw = gradient(x_, y, y_pred)
    w = w-eta*dw

  return history

def train(x, y, w0, b0):
  col1, col2, col3 = st.columns(3)
  with col1:
    eta = st.number_input('Learning Rate', value=.01, step=.01, max_value=.1, min_value=.0001)
  with col2:
    epochs = st.slider('Epochs', step=100, min_value=100, max_value=5000)
  with col3:
    batch_train = st.toggle('Mini-Batch Training')
    batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=10, step=5)
  
  if not batch_train:
    batch_size = 0
  history = fit(x, y, eta, epochs, batch_size)
  st.write('Weights:', *history['weights'][-1])
  draw_result(x,y,history)
  visualize_loss_surface(x, y, w0, b0, history)

def main():
  st.header('Linear Regression')
  with st.sidebar:
    x,y,w,b = prepare_data()
  
  train(x,y,w,b)

if __name__ == "__main__":
  main()