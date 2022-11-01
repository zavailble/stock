import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# panda read csv file
st.title('AAPL,GOOG,META Stock(2017-2022) Analyze')
df_AAPL = pd.read_csv('AAPL.csv')
df_GOOG = pd.read_csv('GOOG.csv')
df_META = pd.read_csv('META.csv')


# choose company dataset
st.sidebar.markdown(''' ## choose dataset of company''' )
com_filter = st.sidebar.radio('',['Apple','Google','Facebook'])
if com_filter == 'Apple':
    st.markdown(''' ## show the data of AAPL''' )
    options = np.array(df_AAPL['Date']).tolist()
    st.sidebar.markdown('''\n \n \n \n \n''')
    (start_time, end_time) = st.sidebar.select_slider("",options = options,value= ('2022-05-04','2018-06-27',),)
    st.sidebar.write("begin time:",start_time)
    st.sidebar.write("end time:",end_time)
    df_AAPL['Date'] = pd.to_datetime(df_AAPL.Date, format = '%Y-%m-%d')
    df_AAPL.index = df_AAPL['Date']  
    df_AAPL1 = df_AAPL[start_time:end_time]
    st.dataframe(df_AAPL1)
    st.dataframe(df_AAPL1.describe())

    

if com_filter == 'Google':
    st.markdown(''' ## show the data of GOOG''' )
    options = np.array(df_GOOG['Date']).tolist()
    st.sidebar.markdown('''\n \n \n \n \n''')
    (start_time, end_time) = st.sidebar.select_slider("",options = options,value= ('2022-05-04','2018-06-27',),)
    st.sidebar.write("begin time:",end_time)
    st.sidebar.write(" end time :",start_time)
    df_AAPL['Date'] = pd.to_datetime(df_AAPL.Date, format = '%Y-%m-%d')
    df_GOOG.index = df_GOOG['Date']  
    df_GOOG1 = df_GOOG[start_time:end_time]
    st.dataframe(df_GOOG1)
    st.dataframe(df_GOOG1.describe()) 



if com_filter == 'Facebook':
    st.markdown(''' ## show the data of META''' )
    options = np.array(df_META['Date']).tolist()
    st.sidebar.markdown('''\n \n \n \n \n''')
    (start_time, end_time) = st.sidebar.select_slider("",options = options,value= ('2022-05-04','2018-06-27',),)
    st.sidebar.write("begin time:",end_time)
    st.sidebar.write(" end time :",start_time)
    df_AAPL['Date'] = pd.to_datetime(df_AAPL.Date, format = '%Y-%m-%d')
    df_META.index = df_META['Date']  
    df_META1 = df_META[start_time:end_time]
    st.dataframe(df_META1)
    st.dataframe(df_META1.describe())



# choose company basic charts
df_AAPL['daily_ret'] = df_AAPL.Adj_Close.pct_change()
df_GOOG['daily_ret'] = df_GOOG.Adj_Close.pct_change()
df_META['daily_ret'] = df_META.Adj_Close.pct_change()

df_AAPL['Date'] = pd.to_datetime(df_AAPL.Date, format = '%Y-%m-%d')
df_GOOG['Date'] = pd.to_datetime(df_GOOG.Date, format = '%Y-%m-%d')
df_META['Date'] = pd.to_datetime(df_META.Date, format = '%Y-%m-%d')

df_AAPL.index = df_AAPL['Date']  
df_META.index = df_META['Date']  
df_GOOG.index = df_GOOG['Date']  

st.sidebar.markdown('''\n \n \n \n \n \n  \n  \n''')
st.sidebar.markdown(''' ## choose chart of company ''')
com_filter = st.sidebar.multiselect('',['Apple','Google','Facebook'],default = 'Apple')


# show company basic charts
st.header(f'Show the basic plot')
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

   
if 'Apple' in com_filter:   
    df_AAPL.Close.plot(ax = ax0,color = 'blue', legend = True,label = 'AAPL closing price',title = 'Close')
    df_AAPL.Volume.plot(ax = ax1,color = 'blue',legend = True, label = 'AAPL daily turnover volume',title = 'Volume')
    df_AAPL.Close.plot(ax = ax2,color = 'blue',legend = True,label = 'AAPL close',title = 'Close and Adj_close')
    df_AAPL.Adj_Close.plot(ax = ax2,color = 'purple',legend = True,label = 'AAPL adi_close')
    df_AAPL.daily_ret.plot(ax = ax3,color = 'blue',legend = True,label = 'AAPL daily return',title = 'daily_ret')
    


if 'Facebook' in com_filter:
    df_META.Close.plot(ax = ax0,color = 'green', legend = True,label = 'META closing price',title = 'Close')
    df_META.Volume.plot(ax = ax1,color = 'green',legend = True, label = 'META daily turnover volume',title = 'Volume')
    df_META.Close.plot(ax = ax2,color = 'green',legend = True,label = 'META close',title = 'Close and Adj_close')
    df_META.Adj_Close.plot(ax = ax2,color = 'gray',legend = True,label = 'META adi_close')
    df_META.daily_ret.plot(ax = ax3,color = 'green',legend = True,label = 'META daily return',title = 'daily_ret')
    


if 'Google' in com_filter:   
    df_GOOG.Close.plot(ax = ax0,color = 'yellow', legend = True,label = 'GOOG closing price',title = 'Close')
    df_GOOG.Volume.plot(ax = ax1,color = 'yellow',legend = True, label = 'GOOG daily turnover volume',title = 'Volume')
    df_GOOG.Close.plot(ax = ax2,color = 'yellow',legend = True,label = 'GOOG close',title = 'Close and Adj_close')
    df_GOOG.Adj_Close.plot(ax = ax2,color = 'red',legend = True,label = 'GOOG adj_close')
    df_GOOG.daily_ret.plot(ax = ax3,color = 'yellow',legend = True,label = 'GOOG daily return',title = 'daily_ret')
    
ax0.set_ylabel('price')    
ax1.set_ylabel('price')
ax2.set_ylabel('price')
ax3.set_ylabel('price')

st.pyplot(fig0) 
st.pyplot(fig1) 
st.pyplot(fig2) 
st.pyplot(fig3) 


#whether choose total revenue and net income or not
st.sidebar.markdown('''\n \n''')
st.sidebar.markdown('''show total revenue and net income''')
revenue_filter = st.sidebar.radio('',['No','Yes'])
if revenue_filter == 'Yes':
    st.markdown(''' ### the total revenue and net income''')
    fig4, ax4 = plt.subplots()
    company = {'name':['AAPL','GOOG','META'],'industry':['Consumer Electronics','Internet Content & Information','Internet Content & Information'],'total_revenue':[365817,257637,117929],'net_income':[94680,76033,39370]}
    df = pd.DataFrame(company)
    df.total_revenue.plot.bar(ax = ax4,color = 'black',legend = True).set_xticks([0,1,2],['AAPL','GOOG','META'],rotation = 30)
    df.net_income.plot.bar(ax = ax4,color = 'purple',legend = True).set_xticks([0,1,2],['AAPL','GOOG','META'],rotation = 30)   
    st.pyplot(fig4)


st.markdown('''\n \n \n \n \n \n  \n  \n''')
st.markdown('''\n \n \n \n \n \n  \n  \n''')
st.markdown('''\n \n \n \n \n \n  \n  \n''')


#OLS regression about different companys' Close price 
st.markdown('''## OLS regression''')
st.write('1. AAPL & GOOG closing price')
fig5, ax5 = plt.subplots()
y = df_AAPL.Close
x = df_GOOG.Close
x[np.isnan(x)] = 0
x[np.isinf(x)] = 0
y[np.isnan(y)] = 0
y[np.isinf(y)] = 0
model = sm.OLS(y,x)
results = model.fit()
st.write(results.params)
st.write(results.summary())
y_fitted = results.fittedvalues
ax5.legend(loc = 'best') 
ax5.plot(x, y, 'o', label = 'data')
ax5.plot(x, y_fitted, 'r--.',label = 'OLS')
st.pyplot(fig5)



st.write('2. GOOG & META closing price')
fig6, ax6 = plt.subplots()
y = df_GOOG.Close
x = df_META.Close
x[np.isnan(x)] = 0
x[np.isinf(x)] = 0
y[np.isnan(y)] = 0
y[np.isinf(y)] = 0
model = sm.OLS(y,x)
results = model.fit()
st.write(results.params)
st.write(results.summary())
y_fitted = results.fittedvalues
ax6.legend(loc = 'best') 
ax6.plot(x, y, 'o', label = 'data')
ax6.plot(x, y_fitted, 'r--.',label = 'OLS')
st.pyplot(fig6)



st.write('3. AAPL & META closing price')
fig7, ax7 = plt.subplots()
y = df_AAPL.Close
x = df_META.Close
x[np.isnan(x)] = 0
x[np.isinf(x)] = 0
y[np.isnan(y)] = 0
y[np.isinf(y)] = 0
model = sm.OLS(y,x)
results = model.fit()
st.write(results.params)
st.write(results.summary())
y_fitted = results.fittedvalues
ax7.legend(loc = 'best') 
ax7.plot(x, y, 'o', label = 'data')
ax7.plot(x, y_fitted, 'r--.',label = 'OLS')
st.pyplot(fig7)


#ADF unit root Test (AAPL as an example)
st.markdown('''## Augmented Dickey-Fuller test''')
fig8, ax8 = plt.subplots()
ts = df_AAPL.Close
#If the null hypothesis cannot be rejected, the original sequence has a unit root
result = adfuller(ts) 
print(result)
#ADF test was performed after the first difference
ts1 = ts.diff().dropna() 
result = adfuller(ts1)
#show ADF test result
st.write('1. ADF test result:')
st.write(result)
st.write('P-value<0.05,so the time series is stationary.')
#show the timing diagram after the first difference
st.write('2. Timing diagram:')
plt.xticks(rotation = 45) 
ax8.plot(ts1)
st.pyplot(fig8) 


#show company info
company = {
'name':['AAPL','GOOG','META'],
'industry':['Consumer Electronics','Internet Content & Information','Internet Content & Information'],
'total_revenue':[365817,257637,117929],
'net_income':[94680,76033,39370]
}


#show stock risk(loss)
st.markdown('''## Stock risk(loss):''')
df=pd.DataFrame(company)
st.dataframe(df)
st.write(f'At 95% confidence interval,the loss isn\'t exceed——>\nAAPL: {abs(df_AAPL.daily_ret.quantile(0.05)*100):.3f}%\nGOOG: {abs(df_GOOG.daily_ret.quantile(0.05)*100):.3f}%\nMETA: {abs(df_META.daily_ret.quantile(0.05)*100):.3f}%')


#Data analysis conclusion
st.markdown('''## Conclusion:''')
st.write('Question 1: Is there a correlation between the share prices of technology stocks (of different companies)?\n')
st.write('Answer 1: Technology stock prices show the same trend and the correlation is strong.\n')
st.write('Question 2: Is there a correlation between market value[total revenue&net income] and daily turnover?\n')
st.write('Answer 2: According to the bar chart of total revenue,net income and linear chart of daily turnover volume, \nthere is a positive correlation between the company\'s market value and daily turnover volume.')
st.write('Question 3: Is there a correlation between daily return and loss?\n')
st.write('Answer 3: There is a positive relationship between daily return and loss, with riskier firms also having relatively higher daily return.')





