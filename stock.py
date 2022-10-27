
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title('AAPL,GOOG,META Stock(2017-2022) Analyze')
df_AAPL=pd.read_csv('AAPL.csv')
df_GOOG=pd.read_csv('GOOG.csv')
df_META=pd.read_csv('META.csv')

#('1.选择公司')
com_filter = st.sidebar.radio('Choose company',['Apple','Google','Facebook'])
if com_filter == 'Apple':
    st.subheader('shou the data of AAPL')
    options = np.array(df_AAPL['Date']).tolist()
    (start_time, end_time) = st.sidebar.select_slider("choose time period：",options = options,value= ('2022-05-04','2018-06-27',),)
    st.sidebar.write("时间序列开始时间:",end_time)
    st.sidebar.write("时间序列结束时间:",start_time)
    df_AAPL['Date'] = pd.to_datetime(df_AAPL.Date, format = '%Y-%m-%d')
    df_AAPL.index = df_AAPL['Date']  
    df_AAPL1 = df_AAPL[start_time:end_time]
    st.dataframe(df_AAPL1)
    st.dataframe(df_AAPL1.describe())



if com_filter == 'Google':
    st.subheader('shou the data of GOOG')
    options = np.array(df_GOOG['Date']).tolist()
    (start_time, end_time) = st.sidebar.select_slider("choose time period：",options = options,value= ('2022-05-04','2018-06-27',),)
    st.sidebar.write("时间序列开始时间:",end_time)
    st.sidebar.write("时间序列结束时间:",start_time)
    df_GOOG['Date'] = pd.to_datetime(df_GOOG.Date, format = '%Y-%m-%d')
    df_GOOG.index = df_GOOG['Date']  
    df_GOOG1 = df_GOOG[start_time:end_time]
    st.dataframe(df_GOOG1)
    st.dataframe(df_GOOG1.describe())

if com_filter == 'Facebook':
    # df_META.Close.plot(legend=True)
    st.subheader('shou the data of META')
    options = np.array(df_META['Date']).tolist()
    (start_time, end_time) = st.sidebar.select_slider("choose tme period：",options = options,value= ('2022-05-04','2018-06-27',),)
    st.sidebar.write("时间序列开始时间:",end_time)
    st.sidebar.write("时间序列结束时间:",start_time)
    df_META['Date'] = pd.to_datetime(df_META.Date, format = '%Y-%m-%d')
    df_META.index = df_META['Date']  
    df_META1 = df_META[start_time:end_time]
    st.dataframe(df_META1)
    st.dataframe(df_META.describe())


df_AAPL['daily_ret']=df_AAPL.Adj_Close.pct_change()
df_GOOG['daily_ret']=df_GOOG.Adj_Close.pct_change()
df_META['daily_ret']=df_META.Adj_Close.pct_change()


com_filter = st.sidebar.multiselect('Choose company',['Apple','Google','Facebook'])
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

if 'Apple' in com_filter:   
    df_AAPL.Close.plot(ax = ax0,color = 'blue', legend = False)
    df_AAPL.Volume.plot(ax = ax1,color = 'blue',legend = True)
    df_AAPL.Close.plot(ax = ax2,color='blue',legend=True)
    df_AAPL.Adj_Close.plot(ax = ax2,color='purple',legend=True)
    df_AAPL.daily_ret.plot(ax = ax4,color='blue',legend=True)
if 'Google' in com_filter:
    df_GOOG.Close.plot(ax = ax0,color = 'green', legend = True)
    df_GOOG.Volume.plot(ax = ax1,color = 'green',legend = True)
    df_GOOG.Close.plot(ax = ax2,color='green',legend=True)
    df_GOOG.Adj_Close.plot(ax= ax2,color='gray',legend=True)
    df_GOOG.daily_ret.plot(ax = ax4,color='green',legend=True)
if 'Facebook' in com_filter:   
    df_META.Close.plot(ax = ax0,color = 'yellow', legend = True)
    df_META.Volume.plot(ax = ax1,color = 'yellow',legend = True)
    df_META.Close.plot(ax = ax2,color='yellow',legend=True)
    df_META.Adj_Close.plot(ax= ax2,color='red',legend=True)
    df_META.daily_ret.plot(ax = ax4, color='yellow',legend=True)
st.pyplot(fig0) 
st.pyplot(fig1) 
st.pyplot(fig2) 
st.pyplot(fig3) 
st.pyplot(fig4) 

company={
    'name':['AAPL','GOOG','META'],
'industry':['Consumer Electronics','Internet Content & Information','Internet Content & Information'],
'total_revenue':[365817,257637,117929],
'net_income':[94680,76033,39370]
}
df=pd.DataFrame(company)
st.dataframe(df)

st.write(f'At 95% confidence interval,the loss isn\'t exceed.\nAAPL: {abs(df_AAPL.daily_ret.quantile(0.05)*100):.3f}%\nGOOG: {abs(df_GOOG.daily_ret.quantile(0.05)*100):.3f}%\nMETA: {abs(df_META.daily_ret.quantile(0.05)*100):.3f}%')








# Median_house_pricing_filter = st.slider('Median_house_pricing_filter:', 0, 500001, 200000)
# df = df[df.median_house_value >= Median_house_pricing_filter]

# location_filter = st.sidebar.multiselect('Choose the location type',df.ocean_proximity.unique(), df.ocean_proximity.unique())
# df = df[df.ocean_proximity.isin(location_filter)]

# income_filter = st.sidebar.radio('Choose the income level',['Low','Medium','High'])
# if income_filter == 'Low':
#     df = df[df.median_income <= 2.5]
# if income_filter == 'Medium':
#     df = df[(df.median_income >= 2.5) & (df.median_income <= 4.5)]
# if income_filter == 'High':
#     df = df[df.median_income <= 4.5]        

# st.subheader('See more filters in the sidebar:')
# st.map(df)

# st.subheader('Histogram of the Median House Value')
# fig, ax = plt.subplots(figsize=(15, 10))
# val = df.median_house_value.hist(bins=30)
# st.pyplot(fig)