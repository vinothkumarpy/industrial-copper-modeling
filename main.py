import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import streamlit as st
import re
from streamlit_lottie import st_lottie
import json as js 
import pickle
import requests
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

selected = option_menu(
    menu_title="Industrial Copper Modeling Application",
    options=['Home', 'Regression and Classification Status'],
    icons=['mic-fill', 'cash-stack', 'phone-flip', "handshake"],
    menu_icon='alexa',
    default_index=0,
)


def lottie(filepath):

    with open(filepath, 'r') as file:

        return js.load(file)

        
if selected == "Regression and Classification Status":

    tab1, tab2  = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"]) 

    with tab1:
            
            st.markdown( f"<h1 style='font-size: 45px;'><span style='color: #00BFFF;'> ML Regression which predicts</span><span style='color: white;'> continuous variable ‘Selling_Price’. </h1>",unsafe_allow_html=True)   

            # Define the possible values for the dropdown menus
            status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
        
            item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
            
            country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
            
            application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
            
            product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                        '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                        '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                        '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                        '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

            # Define the widgets for user input
            with st.form("Regression"):

                col1,col2,col3=st.columns([5,2,5])
                
                with col1:

                    st.write(' ')

                    status = st.selectbox("Status", status_options,key=1)
                    item_type = st.selectbox("Item Type", item_type_options,key=2)
                    country = st.selectbox("Country", sorted(country_options),key=3)
                    application = st.selectbox("Application", sorted(application_options),key=4)
                    product_ref = st.selectbox("Product Reference", product,key=5)
                
                with col3:   

                    st.write( f'<h5 style="color:#00BFFF;">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True )
                    quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                    thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                    width = st.text_input("Enter width (Min:1, Max:2990)")
                    customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                    submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                    
                    st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                    
                            background-color: #00BFFF;
                            color: white;
                            width: 100%;
                        }
                        </style>
                    """, unsafe_allow_html=True)
        
                flag=0 
                
                pattern = "^(?:\\d+|\\d*\\.\\d+)$"

                for i in [quantity_tons,thickness,width,customer]:  

                    if re.match(pattern, i):

                        pass

                    else:  

                        flag=1  

                        break
                
            if submit_button and flag==1:

                if len(i)==0:

                    st.write("please enter a valid number space not allowed")

                else:

                    st.write("You have entered an invalid value: ",i)  
                
            if submit_button and flag==0:
                

                with open(r"D:\vk_project\industrial_copper_modeling\model.pkl", 'rb') as file:

                    loaded_model = pickle.load(file)

                with open(r'D:\vk_project\industrial_copper_modeling\scaler.pkl', 'rb') as f:

                    scaler_loaded = pickle.load(f)

                with open(r"D:\vk_project\industrial_copper_modeling\t.pkl", 'rb') as f:

                    t_loaded = pickle.load(f)

                with open(r"D:\vk_project\industrial_copper_modeling\s.pkl", 'rb') as f:

                    s_loaded = pickle.load(f)

                new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
                new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
                new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
                new_sample1 = scaler_loaded.transform(new_sample)
                new_pred = loaded_model.predict(new_sample1)[0]
                
                font_size = '45px'

                st.markdown(f"<h1 style='font-size: {font_size}; color: #00BFFF;'>Predicted selling price: {np.exp(new_pred)}</h1>", unsafe_allow_html=True)

                
    with tab2: 

            st.markdown( f"<h1 style='font-size: 45px;'><span style='color: #00BFFF;'>ML Classification which predicts </span><span style='color: white;'>win or loss status</h1>",unsafe_allow_html=True)   
        
            with st.form("Classification"):

                col1,col2,col3=st.columns([5,1,5])

                with col1:

                    cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                    cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                    cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                    ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                    cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
                
                with col3: 

                    st.write(' ')

                    citem_type = st.selectbox("Item Type", item_type_options,key=21)
                    ccountry = st.selectbox("Country", sorted(country_options),key=31)
                    capplication = st.selectbox("Application", sorted(application_options),key=41)  
                    cproduct_ref = st.selectbox("Product Reference", product,key=51)           
                    csubmit_button = st.form_submit_button(label="PREDICT STATUS")
        
                cflag=0 

                pattern = "^(?:\\d+|\\d*\\.\\d+)$"
                for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:    

                    if re.match(pattern, k):

                        pass

                    else: 

                        cflag=1  

                        break
                
            if csubmit_button and cflag==1:

                if len(k)==0:

                    st.write("please enter a valid number space not allowed")

                else:

                    st.write("You have entered an invalid value: ",k)  
                
            if csubmit_button and cflag==0:

                with open(r"D:\vk_project\industrial_copper_modeling\cmodel.pkl", 'rb') as file:

                    cloaded_model = pickle.load(file)

                with open(r'D:\vk_project\industrial_copper_modeling\cscaler.pkl', 'rb') as f:

                    cscaler_loaded = pickle.load(f)

                with open(r"D:\vk_project\industrial_copper_modeling\ct.pkl", 'rb') as f:

                    ct_loaded = pickle.load(f)

                # Predict the status for a new sample
                # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
                new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
                new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
                new_sample = cscaler_loaded.transform(new_sample)
                new_pred = cloaded_model.predict(new_sample)
               
                if new_pred==1:

                    st.write('## :green[The Status is Won] ')

                else:

                    st.write('## :red[The status is Lost] ')
                    
    st.markdown( f'<h6 style="color:#00BFFF;">App Created by vinoth kumar</h6>', unsafe_allow_html=True )

if selected == "Home":

    def load_lottieurl(url):

        r = requests.get(url)

        if r.status_code != 200:

            return None
        
        return r.json()

    # Use local CSS
    def local_css(file_name):

        with open(file_name) as f:

            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css(r"D:\vk_project\Air_bnb\style.css")

    lottie_coding = lottie(r"D:\vk_project\lottiite animation\intro vk.json")


    # ---- HEADER SECTION -----``
    with st.container():

        col1,col2=st.columns(2)

        with col1:

            st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> Hi,  </span><span style='color: white;'> I am vinoth kumar </h1>",unsafe_allow_html=True)
            
            st.markdown(
                f"<h1 style='font-size: 40px;'><span style='color: white;'>A Data Scientist,</span><span style='color: #00BFFF;'> From India</span></h1>",
                unsafe_allow_html=True
                )
            
            st.write(f'<h1 style="color:#B0C4DE; font-size: 20px;">A data scientist skilled in extracting actionable insights from complex datasets, adept at employing advanced analytics and machine learning techniques to solve real-world problems. Proficient in Python, statistical modeling, and data visualization, with a strong commitment to driving data-driven decision-making.</h1>', unsafe_allow_html=True)    

            st.write("[view more projects >](https://github.com/vinothkumarpy?tab=repositories)")

        with col2:

            st_lottie(lottie_coding, height=400, key="coding")    

    # ---- WHAT I DO ----
    with st.container():

        st.write("---")

        col1,col2,col3=st.columns(3)

        with col1:

            file = lottie(r"D:\vk_project\lottiite animation\speak with ropot.json")
            st_lottie(file,height=500,key=None)

        with col2:

            st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> WHAT  </span><span style='color: white;'> I DO </h1>",unsafe_allow_html=True)
        
        with col3:

            file = lottie(r"D:\vk_project\lottiite animation\ml process.json")
            st_lottie(file,height=500,key=None)    

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'>Data  </span><span style='color: white;'>Preprocessing:</h1>",unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Gain a deep understanding of dataset variables and types.Handle missing data with appropriate strategies.Prepare categorical features through encoding and data type conversion.Address skewness and ensure data balance.Identify and manage outliers.Resolve date discrepancies for data integrity.</h1>', unsafe_allow_html=True)
    

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Exploratory Data Analysis  </span><span style='color: white;'>(EDA) and Feature Engineering</h1>",unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Visualize and correct skewness.Identify and rectify outliers.Feature improvement and creation for more effective modeling.</h1>', unsafe_allow_html=True) 

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Classi</span><span style='color: white;'>fication</h1>",unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Success and Failure Classification: Focusing on Won and Lost status.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Algorithm Assessment: Evaluating algorithms for classification.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Algorithm Selection: Choosing the Random Forest Classifier.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Hyperparameter Tuning: Fine-tuning with GridSearchCV and cross-validation.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Model Accuracy and Metrics: Assessing performance and metrics.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Model Persistence: Saving the model for future use.</h1>', unsafe_allow_html=True)


        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Regr</span><span style='color: white;'>ession</h1>",unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Algorithm Assessment: Identifying algorithms for regression.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Algorithm Selection: Opting for the Random Forest Regressor.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Hyperparameter Tuning: Fine-tuning with GridSearchCV and cross-validation.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Model Accuracy and Metrics: Evaluating regression model performance.</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">- Model Persistence: Saving the regression model for future applications.</h1>', unsafe_allow_html=True)


        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Interactive  </span><span style='color: white;'>Streamlit UI</h1>",unsafe_allow_html=True)
        st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">Crafted an engaging and user-friendly interface for seamless data exploration and presentation.</h1>', unsafe_allow_html=True)

        st.markdown("[🔗 GitHub Repo >](https://github.com/vinothkumarpy/industrial-copper-modeling.git)")    



    with st.container():

        st.write("---")

        st.markdown( f"<h1 style='font-size: 40px;'><span style='color: #00BFFF;'> Used-Tech  </span><span style='color: white;'>& Skills</h1>",unsafe_allow_html=True)

        col1,col2,col3 =st.columns(3)

        with col1:

            file = lottie(r"D:\vk_project\lottiite animation\python.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>python</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

            file = lottie(r"D:\vk_project\lottiite animation\pandas.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Pandas</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)
            

            file = lottie(r"D:\vk_project\lottiite animation\data_exploaration.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Data Exploaration</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

        with col2:

            file = lottie(r"D:\vk_project\lottiite animation\numpy.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Numpy</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)


            file = lottie(r"D:\vk_project\lottiite animation\Data cleaning.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Data Cleaning</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

            file = lottie(r"D:\vk_project\lottiite animation\working with data set.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Scikit-Learn</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)
            

        with col3: 
            
            file = lottie(r"D:\vk_project\lottiite animation\ml train.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Mechine-Learning Model (regression and classification)</h1>", unsafe_allow_html=True)
            st_lottie(file,height=500,key=None)


            file = lottie(r"D:\vk_project\lottiite animation\frame work.json")
            st.markdown("<h1 style='color: #00BFFF; text-align: center; font-size: 30px;'>Web application development with Streamlit</h1>", unsafe_allow_html=True)
            st_lottie(file,height=200,key=None)

    # ---- PROJECTS ----
    with st.container():

        st.write("---")
        st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> About  </span><span style='color: white;'> Projects </h1>",unsafe_allow_html=True)
        
        col1,col2=st.columns(2)

        with col1:

            file = lottie(r"D:\vk_project\lottiite animation\melting.json")
            st_lottie(file,height=400,key=None)

        with col2:

            st.write("##")
            st.write(f'<h1 style="color:#B0C4DE; font-size: 30px;">The project Enhance data analysis and machine learning skills in the Industrial Copper Modeling project. Tackle complex sales data challenges, employ regression models for pricing predictions, and master lead classification for targeted customer solutions.</h1>', unsafe_allow_html=True)
        
    # ---- CONTACT ----
    with st.container():

        st.write("---")
        st.markdown( f"<h1 style='font-size: 70px;'><span style='color: #00BFFF;'> Get In Touch  </span><span style='color: white;'> With Me </h1>",unsafe_allow_html=True)
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/vinoharish8799@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit" style="background-color: #00BFFF; color: white;">Send</button>
        </form>
        """

        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column: 
            st.empty()    
