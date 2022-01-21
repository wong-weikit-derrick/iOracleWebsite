import streamlit as st
import pandas as pd
import requests
import datetime as dt
import ta
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt

st.title('iOracle')
st.subheader('Predicting the Future Prices of Apple Stock with Machine Learning')
st.markdown('''
            This Data Science project illustrates how an Ensemble Method with LSTM and Random Forest 
            was used to predict the 5-Day future price of Apple's stock based on the past 30 days worth of data.
            ''')



# sidebar choices
side = st.sidebar

side.markdown('## Choose the stock to predict')
stock = side.selectbox("Stock", ("Apple",))

# can add more in future
ticker_dict  = {'Apple': 'aapl'}

# get api
api_url = "https://ioracle-dh3l3t4ama-ew.a.run.app/predict"

### might need to change 
params = {"ticker_name": ticker_dict.get(stock)}

response = requests.get(
    api_url,
    params=params).json()

# compare_df = st.dataframe(response[0])

# pred_df = st.dataframe(response[1])
compare_df = pd.DataFrame(response[0])
compare_df.index = pd.to_datetime(compare_df.index)
compare_df.columns = ['prediction', 'actual']

pred_df = pd.DataFrame(response[1])
# final_df = compare_df.append(pred_df)

# Fix index and join compare_df and pred_df
pred_df.index = [compare_df.index[n]+dt.timedelta(days=7) for n in range(-5, 0)]
final_df = compare_df.append(pred_df)

# Create bollinger bands
hband = BollingerBands(final_df['prediction'], window=5).bollinger_hband()
mband = BollingerBands(final_df['prediction'], window=5).bollinger_mavg()
lband = BollingerBands(final_df['prediction'], window=5).bollinger_lband()

bb_series = pd.DataFrame({'hband':hband, 'mband':mband, 'lband':lband})

final_df = final_df.merge(bb_series, left_index=True, right_index=True)

# Plot actual vs predict graph
plt.figure(figsize=(14, 5))
plt.plot(final_df['hband'], '--', color='green')
plt.plot(final_df['lband'], '--', color='green')
plt.plot(final_df['actual'], label='actual')
plt.plot(final_df['prediction'], label='prediction')
plt.fill_between(final_df.index, final_df['hband'], final_df['lband'], alpha=0.2, color='green')
plt.legend()
plt.title('Model Predictions: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
fig = plt.gcf()
# Plot error
# compare_df['error'] = compare_df['prediction'] - compare_df['actual']
# plt.plot(compare_df['error'], label='error')
# plt.legend()
st.pyplot(fig)


st.metric("MAE in dollars", f"{round(response[2],2)}")
