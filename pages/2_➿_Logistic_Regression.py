import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from stqdm import stqdm
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
  return (X*(y_pred-y).reshape(-1,1)).sum(axis=0)

def back_propagation(w, dw, eta):
  return w-eta*dw

def generate_weights(n_features):
  return np.random.rand(n_features+1)

@st.cache_data
def fit(X, y, ETA, EPOCHS, batch_size=0):
  w = generate_weights(X.shape[1])
  history = {'loss':[], 'accuracy':[], 'weights':[]}
  X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
  for i in stqdm(range(EPOCHS)):
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

def draw_result(X, y, history, history_batch, threshold):
  fig = make_subplots(rows=1, cols=2, subplot_titles=('Decision Boundary', 'History'))
  fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color=np.where(y==0,'red','blue')),
                           text=np.where(y==0,'Class 0','Class 1')), row=1, col=1)
  x1 = np.array([X[:, 0].min()-.05, X[:, 0].max()+.05])
  w = history['weights'][-1]
  x2 = -(x1*w[0] + w[2])/w[1]
  fig.add_trace(go.Scatter(x=x1, y=x2, mode='lines', name = 'Decision Boundary',
                           marker=dict(color='yellowgreen')), row=1, col=1)
  if threshold != .5:
    x2_t = -(x1*w[0] + w[2] + np.log(1/threshold-1))/w[1]
    fig.add_trace(go.Scatter(x=x1, y=x2_t, mode='lines', name = f'Threshold {threshold}',
                             line=dict(color='red', dash='dash')), row=1, col=1)

  if history_batch:
    fig.add_trace(go.Scatter(y=history_batch['loss'], name='Mini-batch Loss'), row=1, col=2)
  fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Batch Loss', line = dict(color='magenta')), row=1, col=2)
  fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Accuracy'), row=1, col=2)

  fig.update_xaxes(title_text="x1", row=1, col=1)
  fig.update_yaxes(title_text="x2", row=1, col=1)
  fig.update_xaxes(title_text="Epochs", row=1, col=2)
  fig.update_layout(showlegend=False)

  st.plotly_chart(fig)

def train(X, y, eta, epochs, batch_size=0):
  t = time.time()
  history = fit(X, y, eta, epochs, batch_size)
  t = (time.time() - t)*1000
  return history, int(t)

def show_result(X, y, history, history_batch, threshold):
  with st.spinner('Visualizing...'):
    draw_result(X, y, history, history_batch, threshold)
    col4, col5 = st.columns(2)
    with col4:
      w = history['weights'][-1]
      X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
      y_pred = feed_forward(X_, w)
      y_pred_label = [0 if i < threshold else 1 for i in y_pred]
      cm = confusion_matrix(y, y_pred_label)
      fig = plt.figure()
      sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
      st.pyplot(fig)
    with col5:
      st.subheader('Classification Report')
      st.text(classification_report(y, y_pred_label))

def main():
  st.header('Logistic Regression')
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    n_samples = st.number_input('Number of Samples', value=200, min_value=100, max_value=10000, step=100)
  with col2:
    eta = st.number_input('Learning Rate', max_value=.1, value=.01)
  with col3:
    epochs = st.number_input('Epochs', value=300, step=50, min_value=10)
  with col4:
    batch_train = st.toggle('Mini-Batch GD')
    batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=20, step=5)
    if not batch_train:batch_size = 0
  threshold = st.slider('Threshold', min_value=.01, max_value=.99, value=.5, step=.01)

  X,y = create_dataset(n_samples)
  history, t = train(X, y, eta, epochs)
  with st.expander('Training Info'):
    st.write('Batch training time:', t, 'ms. Accuracy:', round(history['accuracy'][-1]*100,2), 'Loss:', round(history['loss'][-1],4))
    history_batch = None
    if batch_train:
      history_batch, t_batch = train(X, y, eta, epochs, batch_size)
      w = history_batch['weights'][-1]
      X_ = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
      y_pred = feed_forward(X_, w)
      loss = round(bce_loss(y, y_pred),4)
      acc = round(accuracy(y, y_pred)*100,2)
      st.write('Mini-batch training time:', t_batch, 'ms. Accuracy:', acc, 'Loss:', loss)

  show_result(X, y, history, history_batch, threshold)

if __name__ == "__main__":
  main()