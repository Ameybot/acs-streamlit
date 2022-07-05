
from matplotlib.axis import YAxis
from requests import session
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from streamlit.legacy_caching import clear_cache
import joblib
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap
import copy
import plotly.express as px
from pandas.api.types import is_numeric_dtype

st.set_page_config(
    page_title="Alternative Credit Scoring System",
    page_icon="$",
    layout = 'wide',
)



features = ['UTILITIES_PROCEED', 'JobsReported', 'Assets - Total', 'Cash',
       'Debt in Current Liabilities - Total', 'Long-Term Debt - Total',
       'Gross Profit (Loss)', 'Liabilities - Total', 'EBIT/REV',
       'Interest and Related Expense - Total', 'Inventories - Total', 'Attr1',
       'Attr2', 'Attr3', 'Attr4', 'Attr7', 'Attr9', 'Attr12', 'Attr14',
       'Attr17', 'Attr18', 'Attr19', 'Attr20', 'Attr23', 'Attr28', 'Attr29',
       'Attr30', 'Attr31', 'Attr32', 'Attr33', 'Attr34', 'Attr36', 'Attr40',
       'Attr44', 'Attr45', 'Attr46', 'Attr47', 'Attr50', 'Attr51', 'Attr52',
       'working_capital', 'Attr56', 'Attr60', 'Attr61', 'Attr64',
       'fixed_assets', 'current_assets', 'sales_or_revenue', 'net_profit',
       'cost_of_goods_sold', 'receivables', 'EBIT',
       'operating_expenses', 'LMIIndicator', 'BusinessType',
       'RuralUrbanIndicator', 'Veteran', 'NonProfit', 'BorrowerState',
       'BusinessAgeDescription', 'HubzoneIndicator', 'NAICSCode']


amey_features = ['LMIIndicator', 'BusinessType', 'UTILITIES_PROCEED',
       'RuralUrbanIndicator', 'Veteran', 'JobsReported', 'NonProfit',
       'BorrowerState', 'Ethnicity', 'Race', 'BusinessAgeDescription',
       'HubzoneIndicator', 'Gender', 'NAICSCode', 'Assets - Total', 'Cash',
       'Debt in Current Liabilities - Total', 'Long-Term Debt - Total',
       'Gross Profit (Loss)', 'Liabilities - Total', 'EBIT/REV',
       'Interest and Related Expense - Total', 'Inventories - Total', 'Attr1',
       'Attr2', 'Attr3', 'Attr4', 'Attr7', 'Attr9', 'Attr12', 'Attr14',
       'Attr17', 'Attr18', 'Attr19', 'Attr20', 'Attr23', 'Attr28', 'Attr29',
       'Attr30', 'Attr31', 'Attr32', 'Attr33', 'Attr34', 'Attr36',
       'Attr40', 'Attr44', 'Attr45', 'Attr46', 'Attr47', 'Attr50',
       'Attr51', 'Attr52', 'Attr56', 'Attr60', 'Attr61',
       'Attr64', 'fixed_assets', 'current_assets', 'sales_or_revenue',
       'net_profit', 'working_capital',
       'cost_of_goods_sold', 'receivables', 'EBIT', 'operating_expenses']



@st.cache(allow_output_mutation=True)
def init_zx():
    zx = dict()
    return zx

zx = init_zx()

if 'key' not in st.session_state:
    st.session_state.key = 0

if 'prog1' not in st.session_state:
    st.session_state.prog1 = 0

def update_prog1():
    st.session_state.prog1 += 1/156

if 'prog2' not in st.session_state:
    st.session_state.prog2 = 0

def update_prog2():
    st.session_state.prog2 += 1/342

def update_key_prev():

    st.session_state.key -= 1

def update_key_next():

    st.session_state.key += 1

    if st.session_state.key == 4:
        clear_cache()
        st.session_state.key = 0
        st.session_state.prog1 = 0
        st.session_state.prog2 = 0

if st.session_state.key == 0:
    st.markdown("<h1 style='text-align: center; color: black;'>Alternative Credit Scoring System Dashboard</h1>", unsafe_allow_html=True)

    df = pd.read_csv('150k_merge_NOGOINGBACKEVER.csv',low_memory=False)
    st.dataframe(df)
    st.markdown("<h2 style='text-align: center; color: black;'>Check distributions for selected feature:</h2>", unsafe_allow_html=True)
    ft = st.selectbox('',df.columns)
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x = ft,color_discrete_sequence=['red'])
        st.plotly_chart(fig)
    with c2:
        fig = px.box(df, y = ft,color_discrete_sequence=['red'])
        st.plotly_chart(fig)

    st.markdown("<h2 style='text-align: center; color: black;'>Check relationship between selected features:</h2>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        f1 = st.selectbox('Feature 1',df.columns,index = 15)
        t1 = is_numeric_dtype(df[f1])

    with c2:
        f2 =  st.selectbox('Feature 2',list(set(df.columns) - set(f1)),index = 13)
        t2 = is_numeric_dtype(df[f2])

    if t1 and t2:
            
        fig = px.scatter(x=df[f1], y=df[f2],trendline='ols', trendline_color_override='darkblue',color_discrete_sequence = ['red'])
        fig.update_layout(
        title="Relationship between {} and {}".format(f1,f2),
        xaxis_title=f1,
        yaxis_title=f2,
        )
        st.plotly_chart(fig,use_container_width=True)

    else:

        if t1 == False:

            fig = px.histogram(x = df[f2],color = df[f1])
            fig.update_layout(
            title="Relationship between {} and {}".format(f1,f2),
            )
            st.plotly_chart(fig,use_container_width=True)

        elif t2 == False:

            fig = px.histogram(x = df[f1],color = df[f2])
            fig.update_layout(
            title="Relationship between {} and {}".format(f1,f2),
            
            
            )
            st.plotly_chart(fig,use_container_width=True)


    st.button('Next',on_click = update_key_next)



elif st.session_state.key == 1 :

    my_bar = st.progress(0)
    print(st.session_state.prog1)
    my_bar.progress(st.session_state.prog1)

    # st.write(zx)
    st.markdown("<h1 style='text-align: center; color: black;'>Alternative Data</h1>", unsafe_allow_html=True)
    st.button('Previous',on_click = update_key_prev)

    # with st.form("my_form_1"):

    c1, c2, c3 = st.columns(3)

    with c1:
        zx['LMIIndicator'] = st.selectbox('LMI Indicator', ['Y','N'],help = 'Description',on_change= update_prog1())
        zx['NonProfit'] = st.selectbox('Non Profit', ['Y','N'],help = 'Description',on_change= update_prog1())
        zx['BusinessAgeDescription'] = st.selectbox('Business Age Description', ['Unanswered', 'Existing or more than 2 years old',
        'New Business or 2 years or less','Startup, Loan Funds will Open Business',
        'Change of Ownership'],help = 'Description',on_change= update_prog1())
        zx['Veteran'] = st.selectbox('Veteran', ['Non-Veteran', 'Unanswered', 'Veteran'],help = 'Description',on_change= update_prog1())

    with c2:
        zx['BusinessType'] = st.selectbox('Business Type', ['Limited  Liability Company(LLC)', 'Non-Profit Organization',
        'Independent Contractors', 'Subchapter S Corporation',
        'Sole Proprietorship', 'Self-Employed Individuals', 'Corporation',
        'Partnership', 'Trust', 'Professional Association',
        'Limited Liability Partnership', '501(c)6 – Non Profit Membership',
        'Single Member LLC', '501(c)3 – Non Profit', 'Joint Venture',
        '501(c)19 – Non Profit Veterans',
        'Employee Stock Ownership Plan(ESOP)', 'Cooperative',
        'Non-Profit Childcare Center', 'Qualified Joint-Venture (spouses)'],help = 'Description',on_change= update_prog1())

        zx['BorrowerState'] = st.selectbox('Borrower State', ['TX', 'WI', 'GA', 'CT', 'MD', 'PA', 'CA', 'AL', 'MS', 'TN', 'MI',
        'MN', 'MO', 'AR', 'IA', 'PR', 'LA', 'FL', 'NY', 'IN', 'RI', 'SC',
        'VA', 'NE', 'NV', 'OR', 'OK', 'OH', 'NJ', 'KY', 'MA', 'WA', 'KS',
        'IL', 'ID', 'CO', 'NC', 'AZ', 'DE', 'MT', 'ND', 'WV', 'UT', 'HI',
        'NH', 'AK', 'ME', 'NM', 'SD', 'GU', 'DC', 'WY', 'VI', 'VT', 'MP',
        'AS'],help = 'Description',on_change= update_prog1())

        zx['HubzoneIndicator'] = st.selectbox('Hubzone Indicator', ['Y','N'],help = 'Description',on_change= update_prog1())

        zx['Race'] = st.selectbox('Race', ['White', 'Unanswered', 'Black or African American', 'Asian',
        'American Indian or Alaska Native',
        'Native Hawaiian or Other Pacific Islander'],help = 'Description',on_change= update_prog1())

    with c3:

        zx['RuralUrbanIndicator'] = st.selectbox('Rural Urban Indicator', ['R','U'],help = 'Description',on_change= update_prog1())
        zx['Ethnicity'] = st.selectbox('Ethnicity', ['Not Hispanic or Latino', 'Unknown/NotStated',
        'Hispanic or Latino'],help = 'Description',on_change= update_prog1())
        zx['Gender'] = st.selectbox('Gender', ['Male Owned', 'Unanswered', 'Female Owned'],help = 'Description',on_change= update_prog1())
        zx['NAICSCode'] = st.selectbox('NAICS Code', [62, 81, 48, 72, 56, 53, 54, 11, 23, 44, 33, 71, 61, 21, 42, 32, 52,
            51, 45, 92, 99, 31, 49, 55, 22],help = 'Description',on_change= update_prog1())

    # submitted = st.form_submit_button("Next",on_click = update_key_next)

    st.button('Next',on_click = update_key_next)



elif st.session_state.key == 2:
    
    my_bar = st.progress(0)
    print(st.session_state.prog2)
    my_bar.progress(st.session_state.prog2)

    # st.write(zx)
    st.markdown("<h1 style='text-align: center; color: black;'>Financial Data</h1>", unsafe_allow_html=True)
    st.button('Previous',on_click = update_key_prev)

    # with st.form("my_form_2"):

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # initialInvestment = st.text_input("Starting capital",value=500)
        zx['Interest and Related Expense - Total'] = st.number_input('Interest and Related Expense - Total' ,value=1000,on_change= update_prog2())
        zx['cost_of_goods_sold'] = st.number_input('Cost of Goods Sold',value=1000,on_change= update_prog2())
        zx['EBIT'] = st.number_input('EBIT',value=1000,on_change= update_prog2())
        zx['receivables'] = st.number_input('Receivables',value=1000,on_change= update_prog2())

    with c2:
        # monthlyContribution = st.text_input("Monthly contribution (Optional)",value=100000)
        zx['Cash'] = st.number_input('Cash',value=1000,on_change= update_prog2())
        zx['operating_expenses'] = st.number_input('Operating Expenses',value=1000,on_change= update_prog2())
        zx['Inventories - Total'] = st.number_input('Inventories - Total',value=1000,on_change= update_prog2())
        zx['sales_or_revenue'] = st.number_input('Sales',value=1000,on_change= update_prog2())
        zx['Assets - Total'] = st.number_input('Assets - Total',value=1000,on_change= update_prog2())

    with c3:
        # annualRate = st.text_input("Annual increase rate in percentage",value="15")
        zx['Gross Profit (Loss)'] = st.number_input('Gross Profit (Loss)',value=1000,on_change= update_prog2())
        zx['net_profit'] = st.number_input('Net Profit',value=1000,on_change= update_prog2())
        zx['Debt in Current Liabilities - Total'] = st.number_input('Debt in Current Liabilities - Total',value=1000,on_change= update_prog2())
        zx['Long-Term Debt - Total'] = st.number_input('Long-Term Debt - Total',value=1000,on_change= update_prog2())
        zx['Liabilities - Total'] = st.number_input('Liabilities - Total',value=1000,on_change= update_prog2())
        
    with c4:
        # investingTimeYears = st.text_input("Duration in years:",value=10000)
        zx['fixed_assets'] = st.number_input('Fixed Assets',value=1000,on_change= update_prog2())
        zx['current_assets'] = st.number_input('Current Assets',value=1000,on_change= update_prog2())
        zx['JobsReported'] = st.number_input('Jobs Reported',value=1000,on_change= update_prog2())
        zx['UTILITIES_PROCEED'] = st.number_input('Utilities',value=1000,on_change= update_prog2())

    # submitted = st.form_submit_button("Submit",on_click = update_key_next)
    st.button('Submit',on_click = update_key_next)


elif st.session_state.key == 3:

    working_capital = zx['Assets - Total'] - zx['Liabilities - Total']

    zx['Attr1'] = zx['net_profit'] / zx['Assets - Total']
    zx['Attr2'] = zx['Liabilities - Total'] / zx['Assets - Total']
    zx['Attr3'] = working_capital / zx['Assets - Total']
    zx['Attr4'] = zx['current_assets'] / zx['Debt in Current Liabilities - Total']
    zx['Attr7'] = zx['EBIT'] / zx['Assets - Total']
    zx['Attr9'] = zx['sales_or_revenue'] / zx['Assets - Total']
    zx['Attr12']= zx['Gross Profit (Loss)'] / zx['Debt in Current Liabilities - Total']
    zx['Attr14']= (zx['Gross Profit (Loss)'] + zx['Interest and Related Expense - Total']) / zx['Assets - Total']
    zx['Attr17']= zx['Assets - Total'] / zx['Liabilities - Total']
    zx['Attr18']= zx['Gross Profit (Loss)'] / zx['Assets - Total']
    zx['Attr19']= zx['Gross Profit (Loss)'] / zx['sales_or_revenue']
    zx['Attr20']= (zx['Inventories - Total'] * 365) / zx['sales_or_revenue']
    zx['Attr23']= zx['net_profit'] / zx['sales_or_revenue']
    zx['Attr28']= working_capital / zx['current_assets']
    zx['Attr29']= np.log(zx['Assets - Total'])
    zx['Attr30']= (zx['Assets - Total'] - zx['Cash']) / zx['sales_or_revenue']
    zx['Attr31']= (zx['Gross Profit (Loss)'] + zx['Interest and Related Expense - Total']) / zx['sales_or_revenue']
    zx['Attr32']= (zx['Debt in Current Liabilities - Total'] * 365) / zx['cost_of_goods_sold']
    zx['Attr33']= zx['operating_expenses'] / zx['Debt in Current Liabilities - Total']
    zx['Attr34']= zx['operating_expenses'] / zx['Liabilities - Total']
    zx['Attr36']= zx['sales_or_revenue'] / zx['Assets - Total']
    zx['Attr40']= (zx['current_assets'] - zx['Inventories - Total'] - zx['receivables']) / zx['Debt in Current Liabilities - Total']  
    zx['Attr44']= (zx['receivables'] * 365) / zx['sales_or_revenue']
    zx['Attr45']= zx['net_profit'] / zx['Inventories - Total']
    zx['Attr46']= (zx['current_assets'] - zx['Inventories - Total']) / zx['Debt in Current Liabilities - Total']
    zx['Attr47']= (zx['Inventories - Total'] * 365) / zx['cost_of_goods_sold']
    zx['Attr50']= zx['current_assets'] / zx['Assets - Total']
    zx['Attr51']= zx['Debt in Current Liabilities - Total'] / zx['Assets - Total']
    zx['Attr52']= (zx['Debt in Current Liabilities - Total'] * 365) / zx['cost_of_goods_sold']
    zx['working_capital']= working_capital
    zx['Attr56']= (zx['sales_or_revenue'] - zx['cost_of_goods_sold']) / zx['sales_or_revenue']
    zx['Attr60']= zx['sales_or_revenue'] / zx['Inventories - Total']
    zx['Attr61']= zx['sales_or_revenue'] / zx['receivables']
    zx['Attr64']= zx['sales_or_revenue'] / zx['fixed_assets']
    zx['EBIT/REV'] = zx['EBIT'] /zx['sales_or_revenue']

    # st.write(zx)
    st.button('Previous',on_click = update_key_prev)



    clf = pickle.load(open('xgb_reg_2.pkl', "rb"))



    categorical_cols = ['LMIIndicator', 'BusinessType','RuralUrbanIndicator', 'Veteran', 'NonProfit',
       'BorrowerState', 'BusinessAgeDescription',
       'HubzoneIndicator', 'NAICSCode']  
       
    test_temp = pd.DataFrame()

    for i in features:
        if i not in categorical_cols:
            test_temp[i] = [zx[i]] 
            test_temp[i] = test_temp[i].astype('float64')    

    numeric_cols = list(test_temp.columns)
    numeric_num = len(numeric_cols)

    for i in categorical_cols:
        test_temp[i] = [zx[i]]

    dic = {11: 'Agriculture, Fishing, Forestry, and Hunting',
      21: 'Mining, Quarrying, Oil and Gas Extraction',
      22: 'Utilities', 23: 'Construction', 31:'Manufacturing',
      32: 'Manufacturing', 33: 'Manufacturing', 42: 'Wholesale Trade',
      44: 'Retail Trade', 45:'Retail Trade', 48:'Transport and Warehouse',
      49: 'Transport and Warehouse', 51:'Information',
      52: 'Finance and Insurance', 53:'Real Estate and Rental Leasing',
      54:'Profesisonal, Scientific, and Technical Services',
      55:'Management', 56:'Administrative and Support and Waste Management',
      61:'Educational Services', 62:'Health Care and Social Assistance',
      71: 'Arts, Entertainment and Recreation', 72:'Accomodation and Food Services',
      81: 'Other Services', 92:'Public Administration', 99:'Nonclassifiable Establishments'}

    test_temp['NAICSCode'] = test_temp['NAICSCode'].map(dic)
    # st.write(test_temp)

    # enc = pickle.load(open('onehotencoder.pkl', "rb"))
    enc = joblib.load('encoder_2.joblib')
    enc.handle_unknown = 'ignore'

    # st.write(test_temp[categorical_cols])

    X_enc_cols = enc.transform(test_temp[categorical_cols])

    # st.write(X_enc_cols)

    X_numeric = test_temp.drop(columns=categorical_cols).to_numpy()
    X_numpy = np.hstack((X_numeric, X_enc_cols.toarray()))

    # st.write(X_numpy)

    st.markdown("<h1 style='text-align: center; color: black;'>Credit Report</h1>", unsafe_allow_html=True)

    pred = (clf.predict(X_numpy))[0]


    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = pred+1,
        mode = "gauge+number",
        title = {'text': "Credit Score", 'font':{'size':25}},
        gauge = {'axis': {'range': [1, 9], 'ticks':"",
                'tickmode': 'array', 'tickvals':[1,2,3,4,5,6,7,8],'ticktext':['1','Good','3', 'Fair','5','Poor','7','Very Poor'],
                'tickfont':{'size':15}},
                'steps' : [
                    {'range': [1, 3], 'color': "green"},
                    {'range': [3, 5], 'color': "yellow"},
                    {'range': [5, 7], 'color': "orange"},
                    {'range': [7, 9], 'color': "red"}],
                'bar': {'color': "white"},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 1,
                    'value': pred+1}
                }))

    

    fig.layout.paper_bgcolor = '#f0f2f6'
    st.plotly_chart(fig,use_container_width=True)

    X =  np.array(np.hstack((X_numpy[:, :numeric_num],np.array(enc.inverse_transform(X_numpy[:, numeric_num:])))))
    
    

    # st.write(shap_values_copy)
    

    

    explainer = pickle.load(open('shap_explainer.pkl', "rb"))
    shap_values = explainer(np.array(X_numpy))
    shap_values_copy = copy.deepcopy(shap_values)
    n_categories = []
    n = 0
    x = np.sum(shap_values[:,numeric_num:].values[:,n:(n+test_temp[categorical_cols[0]].nunique())], axis=1).reshape(1,1)
    n += test_temp[categorical_cols[0]].nunique()
    for feat in categorical_cols[1:]:
        x = np.hstack((x,np.sum(shap_values[:,numeric_num:].values[:,n:(n+test_temp[feat].nunique())], axis=1).reshape(1,1)))
        n += test_temp[feat].nunique()
        n_categories.append(n)
    
    shap_values.values = np.hstack((shap_values[:,:numeric_num].values, x))
    shap_values.data = X
    shap_values.feature_names = list(test_temp.columns)

    # st_shap(shap.plots.waterfall(copy.deepcopy(shap_values[0])),height = 300)
    # st_shap(shap.plots.bar(copy.deepcopy(shap_values)),height = 300)
    col = pd.DataFrame(test_temp.columns)
    feat_importances = pd.DataFrame(-1*shap_values.values)
    x= pd.concat([col,feat_importances.T],axis = 1)
    x.columns = ['Feature_names','importance']
    x.sort_values(by = 'importance',ascending = False,inplace = True)
    x = pd.concat([x.head(),x.tail()],axis = 0)
    x.sort_values('importance',inplace = True)
    # st.write(x)
    st.markdown("<h2 style='text-align: center; color: black;'>Feature Importance</h2>", unsafe_allow_html=True)
    layout = go.Layout(autosize=False,
    width=1000,showlegend = True,legend_title_text='Trend',
    height=500, 
        font_family="Monospace",
        font_color=  'black',
        title_font_family="Monospace",
        title_font_color="red",
        legend_title_font_color="black",
        font = dict(size = 18),
        xaxis=dict(title="Values",tickfont = dict(size = 13)),
        yaxis=dict(title="Feature Names",tickfont = dict(size = 13)))
        
    fig = go.Figure(go.Waterfall(orientation = 'h',y = x.Feature_names,x = x.importance),layout = layout)

    st.plotly_chart(fig,use_container_width=True)

    weak_features = list(x.head()['Feature_names'])
    x.sort_values('importance',ascending = False, inplace = True)
    good_features = list(x.head()['Feature_names'])

    c1,c2 = st.columns(2)

    with c1:
    
        with st.expander("Positively contributing features"):
            for pos,value in enumerate(good_features):
                st.write(f'{pos+1}. {value}')

    with c2:
        with st.expander("Negatively contributing features"):
            for pos,value in enumerate(weak_features):
                st.write(f'{pos+1}. {value}')


    st.markdown("<h2 style='text-align: center; color: black;'>Predict Loan Amount (Beta Feature)</h2>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        amt = st.number_input('Loan Amount:' ,value=1000, max_value= 10000)
    with c2:
        term = st.number_input('Term:' ,value=1, max_value= 5)
    allowed = 2000*term/(pred+1)

    if amt<= allowed:
        st.success('Amount Approved')

    else:
        st.warning('Maximum approved amount = ${}'.format(int(allowed)))

    st.button('Reset',on_click = update_key_next)


