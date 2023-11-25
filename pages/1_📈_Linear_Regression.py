import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
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

def visualize_regression_line(x, y, history, history_batch, w_optimal):
  col1,col2 = st.columns(2)
  with col1:
    data = [go.Scatter(y=history['loss'], mode='lines', name='Batch', line = dict(color='magenta'))]
    if history_batch:
      data.append(go.Scatter(y=history_batch['loss'], name='Mini-batch'))
    fig = go.Figure(data=data, layout=go.Layout({'showlegend':False, 'title':'Loss', 'xaxis_title':'Epochs', 'width':300}))
    st.plotly_chart(fig)

  with col2:
    w_ = np.array(history['weights'])
    x_line = np.array([x.min()-.5, x.max()+.5])
    y_gd = x_line * w_[-1,0] + w_[-1,1]
    y_optimal = x_line * w_optimal[0] + w_optimal[1]

    data = [go.Scatter(x=x_line, y=y_gd, mode='lines', name='Batch', line = dict(color='green')), 
            go.Scatter(x=x.flatten(), y=y.flatten(), mode='markers', name='Data'),
            go.Scatter(x=x_line, y=y_optimal, mode='lines', name='Optimal',line = dict(color='red', width=4, dash='dash'))]
    if history_batch:
      w, b = history_batch['weights'][np.argmin(history_batch['loss'])]
      y_gd_batch = x_line * w + b
      data.append(go.Scatter(x=x_line, y=y_gd_batch, mode='lines', name='Mini-batch', line = dict(color='tomato')))

    layout=go.Layout(title="Regression Line", width=500, updatemenus=[dict(type="buttons", buttons=[dict(label=">", method="animate", args=[None])])])
    frames=[go.Frame(data=[go.Scatter(x=x_line, y=x_line*w_[i,0] + w_[i,1], mode='lines', line=dict(color='green'))])
                          for i in range(len(w_))]
    fig = go.Figure(data=data, layout=layout,frames=frames)
    st.plotly_chart(fig)

def visualize_regression_plane(x, y, history, history_batch):
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

@st.cache_data
def visualize_loss_surface(x, y, w_optimal, history):
  w_ = np.array(history['weights'])
  loss_ = np.array(history['loss'])
  if x.shape[1] == 1:
    w0,b0 = w_optimal
    w = np.linspace(min(w0-3, w_[:,0].min()), max(w0+3, w_[:,0].max()), 200)
    b = np.linspace(min(b0-3, w_[:,1].min()), max(b0+3, w_[:,1].max()), 200)
    ww, bb = np.meshgrid(w, b)
    wb = np.c_[ww.ravel(), bb.ravel()]
    loss_surface = np.mean((wb[:,0]*x + wb[:,1]-y)**2, axis=0)
    fig = go.Figure(data=[go.Surface(x=w, y=b, z=loss_surface.reshape(ww.shape)),go.Surface(x=w, y=b, z=loss_surface.reshape(ww.shape))],
                    layout=go.Layout(title='Loss Surface', showlegend=False, hovermode="closest",
                                     scene={'xaxis_title':'w', 'yaxis_title':'b', 'zaxis_title':'Loss'},
                                     updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
                    frames=[go.Frame(data=[go.Scatter3d(x=[w_[i,0]], y=[w_[i,1]], z=[loss_[i]], mode="markers")]) for i in range(len(w_))])
  elif x.shape[1] == 2:
    fig = go.Figure(data=[go.Scatter3d(x=w_[:,0], y=w_[:,1], z=w_[:,2], mode='markers', text=loss_, marker=dict(size=loss_))],
                    layout=go.Layout(title='Loss Surface', showlegend=False, hovermode="closest",
                                     scene={'xaxis_title':'w1', 'yaxis_title':'w2', 'zaxis_title':'b'},
                                     updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
                    frames=[go.Frame(data=[go.Scatter3d(x=[w_[i,0]], y=[w_[i,1]], z=[loss_[i]], mode="markers", marker=dict(size=[loss_[i]]))])
                                     for i in range(len(w_))])
  st.plotly_chart(fig)

def train(x, y, eta, epochs, batch_train, batch_size):
  history, t = fit(x, y, eta, epochs)
  history_batch = None
  if batch_train:
    history_batch, t_batch = fit(x, y, eta, epochs, batch_size)
    w_gd_batch = np.round(history_batch['weights'][np.argmin(history_batch['loss'])], 4)

  x_ = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
  w_optimal = (np.linalg.pinv(x_.T @ x_) @ x_.T @ y).flatten()
  w_gd = np.round(history['weights'][np.argmin(history['loss'])], 4)

  with st.expander('Training Info'):
    st.write('Optimal weights:', *w_optimal.round(decimals=4))
    st.write('Batch GD Weights:', *w_gd, 'Training Time:', t, 'ms')
    if batch_train:st.write('Mini-batch GD Weights:', *w_gd_batch, 'Training Time:', t_batch, 'ms')
  
  return history, history_batch, w_optimal

def visualize_result(x, y, history, history_batch, w_optimal, draw_loss_surface):
  with st.spinner('Visualizing...'):
    if x.shape[1] == 1:
      visualize_regression_line(x, y, history, history_batch, w_optimal)
    elif x.shape[1] == 2:
      visualize_regression_plane(x, y, history, history_batch)
    else:
      draw_loss_history(history, history_batch)
    if draw_loss_surface:
      visualize_loss_surface(x, y, w_optimal, history_batch if history_batch else history)

def main():
  st.header('Linear Regression')
  col1, col2, col3 = st.columns(3)
  with col1:
    n_samples = st.number_input('Number of Samples', value=500, min_value=100, max_value=10000, step=100)
    n_features = st.number_input('Number of Features', value=1, min_value=1, max_value=10, step=1)
    x,y = generate_data(n_samples, n_features)
  with col2:
    eta = st.number_input('Learning Rate', value=.01, step=.01, max_value=.1, min_value=.0001)
    epochs = st.number_input('Epochs', value=100, step=50, min_value=10)
  with col3:
    draw_loss_surface = st.toggle('Draw Loss Surface') if n_features < 3 else False
    batch_train = st.toggle('Mini-Batch GD')
    batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=10, step=5)

  history, history_batch, w_optimal = train(x, y, eta, epochs, batch_train, batch_size)
  visualize_result(x, y, history, history_batch, w_optimal, draw_loss_surface)

if __name__ == "__main__":
  main()