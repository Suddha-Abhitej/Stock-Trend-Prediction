from os import plock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import streamlit as st

start = '2008-01-01'
end = '2021-12-31'

st.title('Stock trend prediction')

user_input = st.text_input('enter stock ','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

st.subheader('Data from 2008 - 2021')
st.write(df.describe())

#visual
st.subheader('Closing Price vs Time chart')
fig = pltfigure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100ma')
ma100 = df.Close.rolling(100).mean()
fig = pltfigure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100ma & 200ma)
ma100 = df.Close.rolling(100).mean()
ma100 = df.Close.rolling(200).mean()
fig = pltfigure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler

# create the scaler object
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


#load  model

model = load_model('keras__model.h5')

#testing
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)


y_pred = model.predict(x_test)

scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_pred = y_pred * scaler_factor
y_test = y_test * scaler_factor

st.subheader('prediction vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_pred,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)