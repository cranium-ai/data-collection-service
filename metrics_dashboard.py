#streamlit run ex_str1.py
import streamlit  as st 
import pandas as pd 
import plotly.express as px

st.header('Metrics Dashboard for Threat intelligence DB \n\n') 

#df = pd.read_csv("Top_3_spotlight_customer.csv")
input_df = pd.read_csv("metrics.csv") 
#select data sources
selected_ds_data = input_df[["source_data","count","last_refresh_date"]].reset_index() 
chart_data = input_df[["source_data","maturity_threat"]].reset_index() 

#Pie Chart
st.subheader('List of Data Sources')
fig=px.pie(selected_ds_data, values = 'count', names='source_data', 
           hover_name='source_data', 
           hover_data=['last_refresh_date'], labels={'last_refresh_date':'Last Updated date'},
           color_discrete_sequence=[ "cyan", "royalblue", "darkblue","lightcyan"])
fig.update_layout(width=600, height=600, paper_bgcolor='#FFFFED', font=dict(color='#383635', size=15))
fig.update_traces(textposition='inside', textinfo='percent+label')
st.write(fig)
st.dataframe(selected_ds_data[["source_data", "last_refresh_date"]],hide_index=True,)


#Donut Chart
st.subheader('List of Data Sources in Threat Intelligence DB')
fig=px.pie(selected_ds_data, values = 'count', names='source_data', hover_name='source_data', 
           hover_data=['last_refresh_date'], labels={'last_refresh_date':'Last Updated date'},
           color_discrete_sequence=px.colors.sequential.RdBu, hole=.3)
fig.update_layout(width=600, height=600, paper_bgcolor='#FFFFED', font=dict(color='#383635', size=15))
fig.update_traces(textposition='inside', textinfo='percent+label')
st.write(fig)


#Bar Chart
st.title(" ")
st.title(" ")
st.subheader("Data Sources by Maturity of threat intelligence")

st.bar_chart(chart_data, x="source_data", y="maturity_threat", color="#90EE90", width=200, height=300)

st.scatter_chart(chart_data, x="source_data", y="maturity_threat", color="#3A5F0B", width=600, height=600, size=600)

#st.line_chart(chart_data, x="source_data", y="maturity_threat", color="#3A5F0B", width=600, height=600)

#streamlit run ex_str1.pymargin=dict(l=0, r=0, t=0, b=0)