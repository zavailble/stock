import streamlit as st
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


st.title('AAPL,GOOG,META Stock(2017-2022) Analyze')
df_AAPL=pd.read_csv('AAPL.csv')
df_GOOG=pd.read_csv('GOOG.csv')
df_META=pd.read_csv('META.csv')

df_AAPL['log_price'] = np.log(df_AAPL['Close'])
df_GOOG['log_price'] = np.log(df_META['Close'])
df_META['log_price'] = np.log(df_META['Close'])

df_AAPL['log_return'] = df_AAPL.log_price.diff()
df_GOOG['log_return'] = df_GOOG.log_price.diff()
df_META['log_return'] = df_META.log_price.diff()

df_AAPL['var'] = np.var(df_AAPL.log_price)
df_GOOG['var'] = np.var(df_GOOG.log_price)
df_META['var'] = np.var(df_META.log_price)

df_AAPL['std'] = np.std(df_AAPL.log_price)
df_GOOG['std'] = np.std(df_GOOG.log_price)
df_META['std'] = np.std(df_META.log_price)

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
    df_AAPL['log_price'] = np.log(df_AAPL.Close)
    df_AAPL['log_return'] = df_AAPL.log_price.diff()
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
    st.dataframe(df_META.describe())











df_AAPL['daily_ret']=df_AAPL.Adj_Close.pct_change()
df_GOOG['daily_ret']=df_GOOG.Adj_Close.pct_change()
df_META['daily_ret']=df_META.Adj_Close.pct_change()

df_AAPL['Date'] = pd.to_datetime(df_AAPL.Date, format = '%Y-%m-%d')
df_GOOG['Date'] = pd.to_datetime(df_GOOG.Date, format = '%Y-%m-%d')
df_META['Date'] = pd.to_datetime(df_META.Date, format = '%Y-%m-%d')

df_AAPL.index = df_AAPL['Date']  
df_META.index = df_META['Date']  
df_GOOG.index = df_GOOG['Date']  

df_AAPL['log_price'] = np.log(df_AAPL['Close'])
df_GOOG['log_price'] = np.log(df_GOOG['Close'])
df_META['log_price'] = np.log(df_META['Close'])

df_AAPL['log_return'] = df_AAPL.log_price.diff()
df_GOOG['log_return'] = df_GOOG.log_price.diff()
df_META['log_return'] = df_META.log_price.diff()

df_AAPL['var'] = np.var(df_AAPL.log_price)
df_GOOG['var'] = np.var(df_GOOG.log_price)
df_META['var'] = np.var(df_META.log_price)

df_AAPL['std'] = np.std(df_AAPL.log_price)
df_GOOG['std'] = np.std(df_GOOG.log_price)
df_META['std'] = np.std(df_META.log_price)

st.sidebar.markdown('''\n \n \n \n \n \n  \n  \n''')
st.sidebar.markdown(''' ## choose chart of company ''')
com_filter = st.sidebar.multiselect('',['Apple','Google','Facebook'],default='Apple')

chart_filter = st.sidebar.radio('Choode the chart:',['Close','Volume','Adj_close','log_price','Daily_ret'])
st.header(f'Show the basic plot ')
fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig6, ax6 = plt.subplots()
ax0.set_ylabel('price')    


if 'Apple' in com_filter:   
    if 'Close' == chart_filter:
        df_AAPL.Close.plot(ax = ax0,color = 'blue', legend = True,label = 'AAPL closing price')
        st.pyplot(fig0) 

    elif 'Volume' == chart_filter:
        df_AAPL.Volume.plot(ax = ax1,color = 'blue',legend = True, label = 'AAPL daily turnover volume')
        st.pyplot(fig1) 

    elif 'Adj_Close' == chart_filter:
        df_AAPL.Close.plot(ax = ax2,color='blue',legend=True,label = 'AAPL close')
        df_AAPL.Adj_Close.plot(ax = ax2,color='purple',legend=True,label = 'AAPL adi_close')
        st.pyplot(fig2)

    elif 'log_Close' == chart_filter:
        df_AAPL.log_price.plot(ax=ax3,color='blue',legend=True,label= 'AAPL log_price')
        st.pyplot(fig3)     

    if 'daily_ret' == chart_filter:    
        df_AAPL.daily_ret.plot(ax = ax4,color = 'blue',legend = True,label = 'AAPL daily return')
        st.pyplot(fig4) 
    # df_AAPL.log_return.plot(ax = ax4,color = 'purple',legend = True,label = 'AAPL log_return')       
    # if 'Google' in com_filter:
    #     ax6=plt.scatter(df_AAPL.log_return, df_GOOG.log_return) 
    #     ax6.set_xlabel('df_AAPL.log_return','df_GOOG.log_return')
        

if 'Facebook' in com_filter:
    
    if 'Close' in chart_filter:
        df_META.Close.plot(ax = ax0,color = 'green', legend = True,label = 'META closing price')
        st.pyplot(fig0) 

    if 'Volume' in chart_filter:
        df_META.Volume.plot(ax = ax1,color = 'green',legend = True, label = 'META daily turnover volume')
        st.pyplot(fig1) 

    if 'Adj_Close' in chart_filter:
        df_META.Close.plot(ax = ax2,color='green',legend=True,label = 'META close')
        df_META.Adj_Close.plot(ax = ax2,color='gray',legend=True,label = 'META adi_close')
        st.pyplot(fig2)

    if 'log_Close' in chart_filter:
        df_META.log_price.plot(ax=ax3,color='green',legend=True,label= 'META log_price')
        st.pyplot(fig3) 

    if 'daily_ret' in chart_filter:    
        df_META.daily_ret.plot(ax = ax4,color = 'green',legend = True,label = 'META daily return')
        st.pyplot(fig4) 



if 'Gooogle' in com_filter:   
    
    if 'Close' in chart_filter:
        df_GOOG.Close.plot(ax = ax0,color = 'yellow', legend = True,label = 'GOOG closing price')
        st.pyplot(fig0) 

    if 'Volume' in chart_filter:
        df_GOOG.Volume.plot(ax = ax1,color = 'yellow',legend = True, label = 'GOOG daily turnover volume')
        st.pyplot(fig1) 

    if 'Adj_Close' in chart_filter:
        df_GOOG.Close.plot(ax = ax2,color='yellow',legend=True,label = 'GOOG close')
        df_GOOG.Adj_Close.plot(ax = ax2,color='red',legend=True,label = 'GOOG adj_close')
        st.pyplot(fig2)

    if 'log_Close' in chart_filter:
        df_GOOG.log_price.plot(ax=ax3,color='yellow',legend=True,label= 'GOOG log_price')
        st.pyplot(fig3) 

    if 'daily_ret' in chart_filter:    
        df_GOOG.daily_ret.plot(ax = ax4,color = 'yellow',legend = True,label = 'GOOG daily return')
        st.pyplot(fig4) 



ax0.set_ylabel('price')    





# fig6, ax6 = plt.subplots()
# ret_all = [df_AAPL.log_return,df_GOOG.log_return,df_META.log_return]
# ax6 = scatter_matrix(ret_all, diagonal='kde', figsize=(10, 10))
# st.pyplot(fig6)
st.sidebar.markdown('''\n \n \n \n \n \n  \n  \n''')

st.sidebar.markdown('''show the revenue and net_income''')
revenue_filter = st.sidebar.radio('',['No','Yes'])
if revenue_filter == 'Yes':
    st.markdown(''' ### the total revenue and net income''')
    fig5, ax5 = plt.subplots()
    company={'name':['AAPL','GOOG','META'],'industry':['Consumer Electronics','Internet Content & Information','Internet Content & Information'],'total_revenue':[365817,257637,117929],'net_income':[94680,76033,39370]}
    df=pd.DataFrame(company)
    df.total_revenue.plot.bar(ax=ax5,color='black',legend=True).set_xticks([0,1,2],['AAPL','GOOG','META'],rotation=30)
    df.net_income.plot.bar(ax=ax5,color='purple',legend=True).set_xticks([0,1,2],['AAPL','GOOG','META'],rotation=30)   
    st.pyplot(fig5)










st.markdown('''\n \n \n \n \n \n  \n  \n''')
st.markdown('''\n \n \n \n \n \n  \n  \n''')
st.markdown('''\n \n \n \n \n \n  \n  \n''')

fig7, ax7 = plt.subplots()
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
ax7.legend(loc='best') 
ax7.plot(x, y, 'o', label='data')
ax7.plot(x, y_fitted, 'r--.',label='OLS')
st.pyplot(fig7)

#ADF单位根检验
st.markdown('''## ADF 单位根检验''')
fig8, ax8 = plt.subplots()
ts = df_AAPL.Close
 #ADF单位根检验
result = adfuller(ts) #不能拒绝原假设，即原序列存在单位根
print(result)
ts1= ts.diff().dropna() #一阶差分再进行ADF检验
result = adfuller(ts1)
st.write(result)
plt.xticks(rotation=45) #坐标角度旋转
ax8.plot(ts1)
st.markdown('''不能拒绝原假设，即原序列存在单位根''')
st.pyplot(fig8) #一阶差分后的时序图

from statsmodels.tsa import stattools #白噪声检验:Ljung-Box检验
LjungBox=stattools.q_stat(stattools.acf(ts1)[1:12],len(ts1))[1] #显示第一个到第11个白噪声检验的p值
st.write(LjungBox)  #检验的p值大于0.05，因此不能拒绝原假设，差分后序列白噪声检验通过
st.write('P-value<0.05,so the time series is stationary.')


company={
    'name':['AAPL','GOOG','META'],
'industry':['Consumer Electronics','Internet Content & Information','Internet Content & Information'],
'total_revenue':[365817,257637,117929],
'net_income':[94680,76033,39370]
}
df=pd.DataFrame(company)
st.dataframe(df)
st.write(f'At 95% confidence interval,the loss isn\'t exceed——>\nAAPL: {abs(df_AAPL.daily_ret.quantile(0.05)*100):.3f}%\nGOOG: {abs(df_GOOG.daily_ret.quantile(0.05)*100):.3f}%\nMETA: {abs(df_META.daily_ret.quantile(0.05)*100):.3f}%')


st.subheader('Conclusion:')
st.write('Question 1: Is there a correlation between company market value (net income) and daily turnover volumn?\n')
st.write('Answer 1: There is a positive relationship between company market value (net income) and daily turnover.\nCompanies with larger net income have higher daily turnover volumn.\n\n')
st.write('Question 2: Is there a correlation between log_return and daily turnover volume?')
st.write('Answer 2: According to OLS (ordinary least squares) regression and ADF stationarity test, there is a positive relationship between log_return and daily turnover volume.')
st.write('Question 3: Is there a correlation between daily return and loss?\n')
st.write('Answer 3: There is a positive relationship between daily return and loss, with riskier firms also having relatively higher daily return.')





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