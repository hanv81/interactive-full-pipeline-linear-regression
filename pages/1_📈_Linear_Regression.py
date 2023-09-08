import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from stqdm import stqdm

@st.cache_data
def generate_data(n_samples, n_features):
  x = np.random.randn(n_samples, n_features)
  x_ = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
  w = np.random.randint(-10,10,n_features + 1)
  noise = np.random.randn(n_samples)
  y = (x_*w).sum(axis=1) + noise
  return x, y.reshape(-1,1)

def prepare_data():
  n_samples = st.number_input('Number of Samples', value=1000, min_value=100, max_value=10000, step=500)
  n_features = st.number_input('Number of Features', value=1, min_value=1, max_value=5, step=1)
  x,y = generate_data(n_samples, n_features)
  return x,y

@st.cache_data
def visualize_loss_surface(x, y, w_optimal, history):
  w_ = np.array(history['weights'])
  loss_ = np.array(history['loss'])
  w0,b0 = w_optimal
  w = np.linspace(w0-3, w0+3, 200)
  b = np.linspace(b0-3, b0+3, 200)
  ww, bb = np.meshgrid(w, b)
  wb = np.c_[ww.ravel(), bb.ravel()]
  loss = np.mean(((wb[:,0]*x + wb[:,1])-y)**2, axis=0)

  fig = go.Figure(data=[go.Surface(x=w, y=b, z=loss.reshape(ww.shape)),
                        go.Scatter3d(x=w_[:,0], y=w_[:,1], z=loss_, mode='markers'),
                        go.Scatter3d(x=w_[[0,-1],0], y=w_[[0,-1],1], z=loss_[[0,-1]], mode='markers')])
  st.plotly_chart(fig)

def mse(y, y_pred):
  return ((y-y_pred)**2).mean()

def feed_forward(x, w):
  return (x@w).reshape(-1,1)

def gradient(x, y, y_pred):
  return 2*(x*(y_pred - y)).mean(axis=0)

def gradient_descent(x, y, w, eta):
  y_pred = feed_forward(x, w)
  loss = mse(y, y_pred)
  dw = gradient(x, y, y_pred)
  w = w-eta*dw
  return w, loss

@st.cache_data
def fit(x, y, eta, epochs, batch_size=0):
  t = time.time()
  w = np.random.rand(x.shape[1]+1)
  history = {'loss':[], 'weights':[]}
  x_ = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
  for _ in stqdm(range(epochs)):
    if batch_size > 0:
      id = np.random.choice(len(y), batch_size)
      xx, yy = x_[id], y[id]
    else:
      xx, yy = x_, y
    history['weights'].append(w)
    w, loss = gradient_descent(xx, yy, w, eta)
    history['loss'].append(loss)

  t = int((time.time() - t)*1000)
  return history, t

def draw_result(x, y, history, history_batch, w_optimal):
  w, b = history['weights'][np.argmin(history['loss'])]
  x_line = np.array([x.min()-.5, x.max()+.5])
  y_gd = x_line * w + b
  y_optimal = x_line * w_optimal[0] + w_optimal[1]

  fig = make_subplots(rows=1, cols=2, subplot_titles=('Regression Line', 'Loss'))
  fig.add_trace(go.Scatter(x=x.flatten(), y=y.flatten(), mode='markers', name='Data'), row=1, col=1)
  fig.add_trace(go.Scatter(x=x_line.flatten(), y=y_optimal.flatten(), mode='lines', name='Optimal',
                           line = dict(color='red', width=4, dash='dash')), row=1, col=1)
  fig.add_trace(go.Scatter(x=x_line.flatten(), y=y_gd.flatten(), mode='lines', name='Batch',
                           line = dict(color='green')), row=1, col=1)
  if history_batch:
    w, b = history_batch['weights'][np.argmin(history_batch['loss'])]
    y_gd_batch = x_line * w + b
    fig.add_trace(go.Scatter(x=x_line.flatten(), y=y_gd_batch.flatten(), mode='lines', name='Mini-batch',
                           line = dict(color='tomato')), row=1, col=1)
    fig.add_trace(go.Scatter(y=history_batch['loss'], name='Mini-batch'), row=1, col=2)

  fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Batch', line = dict(color='magenta')), row=1, col=2)
  fig.update_xaxes(title_text="x", row=1, col=1)
  fig.update_xaxes(title_text="Epochs", row=1, col=2)
  fig.update_yaxes(title_text="y", row=1, col=1)
  
  fig.update_layout(height=500, width=800)
  st.plotly_chart(fig)

def draw_result_3d(x, y, history, history_batch):
  w_batch = history['weights'][np.argmin(history['loss'])]
  x_plane = [x[:,0].min(), x[:,0].max()]
  y_plane = [x[:,1].min(), x[:,1].max()]
  xx, yy = np.meshgrid(x_plane, y_plane)
  xy = np.c_[xx.ravel(), yy.ravel()]
  z_batch = xy[:,0]*w_batch[0] + xy[:,1]*w_batch[1] + w_batch[2]
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Regression Plane', 'Loss'),
                      specs=[[{"type": "scatter3d"}, {"type": "scatter"}]])

  fig.add_trace(go.Scatter3d(x=x[:,0], y=x[:,1], z=y.flatten(), mode='markers', name='Data'), row=1, col=1)
  fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_batch.reshape(xx.shape), name='Batch'), row=1, col=1)

  if history_batch:
    w_minibatch = history_batch['weights'][np.argmin(history_batch['loss'])]
    z_minibatch = xy[:,0]*w_minibatch[0] + xy[:,1]*w_minibatch[1] + w_minibatch[2]
    fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_minibatch.reshape(xx.shape), name='Mini-batch'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history_batch['loss'], name='Mini-batch'), row=1, col=2)
  fig.add_trace(go.Scatter(y=history['loss'], name='Batch', line = dict(color='magenta')), row=1, col=2)

  fig.update_xaxes(title_text="Epochs", row=1, col=2)
  fig.update_layout(height=500, width=800)
  st.plotly_chart(fig)

def draw_loss_history(history, history_batch):
  data = [go.Scatter(y = history_batch['loss'], mode = 'lines', name='Mini-batch')] if history_batch else []
  data.append(go.Scatter(y = history['loss'], mode = 'lines', name='Batch', line = dict(color='magenta')))
  fig = go.Figure(data)
  fig.update_layout(xaxis_title="Epochs", yaxis_title="Loss")
  st.plotly_chart(fig)

def train(x, y, eta, epochs, batch_train, batch_size, draw_loss, show_training_result):
  history, t = fit(x, y, eta, epochs)
  history_batch = None
  if batch_train:
    history_batch, t_batch = fit(x, y, eta, epochs, batch_size)
    w_gd_batch = np.round(history_batch['weights'][np.argmin(history_batch['loss'])], 4)

  x_ = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
  w_optimal = np.linalg.pinv(x_.T @ x_) @ x_.T @ y
  w_gd = np.round(history['weights'][np.argmin(history['loss'])], 4)
  if show_training_result:
    st.write('Optimal weights:', *w_optimal.flatten().round(decimals=4))
    st.write('Batch GD Weights:', *w_gd, 'Training Time:', t, 'ms')
    if batch_train:st.write('Mini-batch GD Weights:', *w_gd_batch, 'Training Time:', t_batch, 'ms')

  with st.spinner('Visualizing...'):
    if x.shape[1] == 1:
      draw_result(x, y, history, history_batch, w_optimal)
      if draw_loss:
        visualize_loss_surface(x, y, w_optimal.flatten(), history_batch if batch_train else history)
    elif x.shape[1] == 2:
      draw_result_3d(x, y, history, history_batch)
    else:
      draw_loss_history(history, history_batch)

def main():
  st.header('Linear Regression')
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    x,y = prepare_data()
  with col2:
    eta = st.number_input('Learning Rate', value=.01, step=.01, max_value=.1, min_value=.0001)
    epochs = st.number_input('Epochs', value=100, step=50, min_value=10)
  with col3:
    batch_train = st.toggle('Mini-Batch GD')
    batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=10, step=5)
  with col4:
    show_training_info = st.toggle('Show Training Info')
    draw_loss = st.toggle('Draw Loss Surface')

  train(x, y, eta, epochs, batch_train, batch_size, draw_loss, show_training_info)

if __name__ == "__main__":
  main()