import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    st.header('Linear Regression')
    uploaded_file = st.sidebar.file_uploader("Upload data file", type='csv')
    
    if uploaded_file is not None:
        model = LinearRegression()
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        features = df.columns[:-1]
        label = df.columns[-1]
        X = df[features].values
        y = df[label].values

        def form_callback():
            if len(options) < 2:
                return
            i = features.get_loc(options[0])
            j = features.get_loc(options[1])

            fig = go.Figure(data=[go.Scatter3d(x=X[:,i], y=X[:,j], z=y, mode='markers')])
            fig.update_layout(scene = dict(
                xaxis_title=features[i],
                yaxis_title=features[j],
                zaxis_title=label),
                width=700, margin=dict(r=20, b=10, l=10, t=10))
            
            if 'x_new' in st.session_state:
                x_new = st.session_state.x_new
                y_new = st.session_state.y_new
                fig.add_scatter3d(x=[x_new[0]], y=[x_new[1]], z=[y_new])

        options = st.multiselect('Select 2 features to visualize', features, [features[0], features[1]], on_change=form_callback)
        if len(options) > 1:
            i = features.get_loc(options[0])
            j = features.get_loc(options[1])

            fig = go.Figure(data=[go.Scatter3d(x=X[:,i], y=X[:,j], z=y, mode='markers')])
            fig.update_layout(scene = dict(
                xaxis_title=features[i],
                yaxis_title=features[j],
                zaxis_title=label),
                width=700, margin=dict(r=20, b=10, l=10, t=10))

            if 'x_new' in st.session_state:
                x_new = st.session_state.x_new
                y_new = st.session_state.y_new
                fig.add_scatter3d(x=[x_new[0]], y=[x_new[1]], z=[y_new])

            # if 'model' in st.session_state:
            #     model = st.session_state.model

            #     x_min, x_max = X[:,i].min() , X[:,i].max() 
            #     y_min, y_max = X[:,j].min() , X[:,j].max() 
            #     xrange = np.linspace(x_min, x_max, 50)
            #     yrange = np.linspace(y_min, y_max, 50)
            #     xx, yy = np.meshgrid(xrange, yrange)

            #     inp = np.c_[xx.ravel(), yy.ravel()]

            #     y_test_pred = model.predict(inp)
            #     y_test_pred = y_test_pred.reshape(xx.shape)

            #     fig.add_traces(go.Surface(x=xrange, y=yrange, z=y_test_pred, name='pred_surface'))

            st.write(fig)
        else:
            fig, ax = plt.subplots()
            plt.scatter(X[:,0], y)
            st.write(fig)
    
        test_size = st.slider('Select test size', 10, 40, 20, 5)
        submitted = st.button("Train")
        if submitted:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            model.fit(X_train, y_train)
            # y_test_pred = model.predict(X_test)
            # mae = mean_absolute_error(y_test, y_test_pred)
            # mse = mean_squared_error(y_test, y_test_pred)
            # model_score = model.score(X_test, y_test)
            s = ''
            for w,f in zip(model.coef_, features):
                w = round(w, 4)
                if w > 0:
                    if s == '':
                        s = str(w) + f
                    else:
                        s += ' + ' + str(w) + f
                else:
                    s += ' - ' + str(abs(w)) + f
            if model.intercept_ > 0:
                s += ' + ' + str(round(model.intercept_, 4))
            else:
                s += ' - ' + str(round(model.intercept_, 4))
            s = label + ' = ' + s
            st.write('Model: ' + s)
            # st.write('MAE: ' + str(round(mae, 4)))
            # st.write('MSE: ' + str(round(mse, 4)))
            # st.write('Model score: ' + str(round(model_score, 4)))

            st.session_state.model = model
            st.session_state.features = features
            st.session_state.x_test = X_test
            st.session_state.df = df

    with st.form("inference form"):
        
        if 'features' in st.session_state:
            x_new = []
            for feature in features:
                f_val = st.slider(feature, 0, 100, 0, 5)
                x_new += [f_val]
            submitted = st.form_submit_button("Predict")
            if submitted:
                model = st.session_state.model
                y_new = model.predict([x_new])[0]
                st.session_state.x_new = x_new
                st.session_state.y_new = y_new
                st.write('Sales = ' + str(round(y_new,4)) + ' USD')

if __name__ == "__main__":
    main()