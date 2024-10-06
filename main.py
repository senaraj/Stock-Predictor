import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

start="2015-1-1"
today=date.today().strftime("%Y-%m-%d")

st.title("Stock Predictor")
st.subheader('Made By Senaraj')

stock = ("NVDA", "AAPL","GOOG","NFLX", "AMZN", "MSFT","GME","BTC-USD")

selected_stocks = st.selectbox("Select Stock for Prediction", stock)
n_years = st.slider("Years of predicion:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load the stock data...")
data = load_data(selected_stocks)
data_load_state.text("loading data...... done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()


df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())
st.write('forecast data')
fig2=plot_plotly(m, forecast)
st.plotly_chart(fig2)

st.write('forecast component')

fig2=m.plot_components(forecast)
st.write(fig2)