from turtle import color
from attr import attr
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from bokeh.plotting import figure
from datetime import datetime
import altair as alt
from numpy import log
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import month_plot, seasonal_plot, plot_acf, plot_pacf, quarter_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import boxcox
import plotly.figure_factory as ff
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#from forecast import generate_forecast, calculate_smape

def main():
    #st.title('Forecast Gold Price')
    st.markdown("<h1 style='text-align: center; color: #15F4F4; text-decoration:underline;'>Bankruptcy Prevention</h1>", unsafe_allow_html=True)
    image = Image.open('bank.jpg')
    st.sidebar.image(image, width=300) 
if __name__ == "__main__":
    main()    

df = pd.read_csv('bankruptcy-prevention.csv',sep=";")

st.sidebar.title("1. Data")
if st.sidebar.checkbox("Display data", False):
    st.subheader("Show 'Gold Price' dataset")
    st.table(df)    


df1 = pd.read_csv("bankruptcy-prevention.csv",sep=";")


st.sidebar.title("2. Options")
page = st.sidebar.selectbox('Page Navigation', [" ","Dataset Information", "Classification Analysis"])


if page=="Dataset Information":
    st.subheader("1. Overview")
    with st.expander("Introduction"):
        st.write("""
        This is a classification project, since the variable to predict is binary (bankruptcy or non-bankruptcy). 
        The goal here is to model the probability that a business goes bankrupt from different features.
         """)
        #st.write(df1.index) 
    st.subheader("2. Description of Dataset") 
    with st.expander("Shape of Dataset"):
        df.shape
    with st.expander("Name of Columns"):
        df.columns
    with st.expander("Null Values"):
        st.write(df.isnull().sum())
    with st.expander("Dataset insights"):
        st.write(df.describe())
    st.subheader("3. Count Plot")    
    with st.expander("Count plot with description"):
        st.write("""
                countplot() method is used to Show the counts of observations in each categorical bin using bars. Parameters : This method is accepting the 
                following parameters that are described below: x, y: This parameter take names of variables in data or vector data, 
                optional, Inputs for plotting long-form data.
         """)
        st.text("This is the count plot :") 
        fig,ax = plt.subplots()
        sns.countplot(x=df[' class'], palette='turbo', linewidth=1)
        st.pyplot(fig)
    st.subheader("4. Count Plot for all features")    
    with st.expander("Count plot"):
        st.write("""
                Here we count values for all features by taking class as a reference.
         """)
        st.text("This is the Distribution Plot :") 
        fig = plt.figure(figsize=(15, 12))
        for i, predictor in enumerate(df.drop(columns = [' class'])):
            ax = plt.subplot(3, 2, i + 1)
            sns.countplot(data=df, x=predictor, hue=' class')
        st.pyplot(fig)

    st.subheader("5. Correlation Graph")    
    with st.expander("Description"):
        st.write("""Correlation measures the relationship, or association, between two variables by looking at how the variables 
        change with respect to each other. Statistical correlation also corresponds to simultaneous changes between two variables, 
        and it is usually represented by linear relationships.""")
        fig = plt.figure()
        sns.heatmap(df.corr(), cmap='Greens', vmin=-1, vmax=1, annot=True, annot_kws={'fontsize':12})
        st.pyplot(fig)


    st.subheader("6. Box Plot")
    with st.expander("Description"):
        st.write('''Box plots are used to show distributions of numeric data values, especially when you want to compare them between
         multiple groups. They are built to provide high-level information at a glance, offering general information about a group of 
         data's symmetry, skew, variance, and outliers.''')
        fig = plt.figure(figsize=(15, 12))
        for i, predictor in enumerate(df.drop(columns = [' class'])):
            ax = plt.subplot(3, 2, i + 1)
            sns.boxplot(data=df, x=' class', y=predictor )
        st.pyplot(fig)

    st.subheader("7. Value Count")
    with st.expander("Options"):
        genre1 = st.radio("Select Features",('None','Industrial risk', 'Management risk', 'Financial flexibility','Credibility','Competitiveness','Operating risk')) 
        if genre1 == 'Industrial risk':
            a = df['industrial_risk'].value_counts()
            st.write(a)
        if genre1 == 'Management risk':
            a = df[' management_risk'].value_counts()
            st.write(a)
        if genre1 == 'Financial flexibility':
            a = df[' financial_flexibility'].value_counts()
            st.write(a)
        if genre1 == 'Credibility':
            a = df[' credibility'].value_counts()
            st.write(a)
        if genre1 == 'Competitiveness':
            a = df[' competitiveness'].value_counts()
            st.write(a)
        if genre1 == 'Operating risk':
            a = df[' operating_risk'].value_counts()
            st.write(a)

    st.subheader("8. Pie chart representation of CLASS feature")
    with st.expander("Description"):
        st.write('''Pie charts show the parts-to-whole relationship.A pie chart is a circle that is divided into areas, or slices. 
        Each slice represents the count or percentage of the observations of a level for the variable.Pie charts are often used in business.''')
        encode = LabelEncoder()
        df[' class'] = encode.fit_transform(df[' class'])
        a =df[' class'].value_counts()[0]     
        b =df[' class'].value_counts()[1]     
        fig1, ax1 = plt.subplots()
        label = ['bankruptcy', 'non-bankruptcy']
        count = [a, b]
        colors = ['skyblue', 'yellowgreen']
        explode = (0, 0.1)  # explode 2nd slice
        plt.pie(count, labels=label, autopct='%0.2f%%', explode=explode, colors=colors,shadow=True, startangle=90)
        st.pyplot(fig1)                                 

              

        
if page=="Classification Analysis":
                                       
    genre = st.sidebar.radio("Select",('None','Performance Metrics used','Evaluation Metrics', 'Predict Values')) 
    if genre =='Performance Metrics used':  
        st.subheader("1. Performance Metrics used :")
        with st.expander("More Info on evaluation Matrics"):
            st.markdown(""" The following metrics can be computed to evaluate model performance:""")
            st.markdown("""* 1.**Accuracy**: Accuracy represents the number of correctly classified data instances over the total number 
                             of data instances.Accuracy may not be a good measure if the dataset is not balanced 
                             (both negative and positive classes have different number of data instances).""")
            st.markdown("""* 2.**Precision**: Precision is one indicator of a machine learning model's performance-the quality of a 
                            positive prediction made by the model. Precision refers to the number of true positives divided by the total number of positive 
                            predictions (i.e., the number of true positives plus the number of false positives).""") 
            st.markdown("""* 3.**Recall**: Recall is how many of the true positives were recalled (found), i.e. 
                              how many of the correct hits were also found.""") 
            st.markdown("""* 4.**F1-score**: F1-score is one of the most important evaluation metrics in machine learning. It elegantly 
                             sums up the predictive performance of a model by combining two otherwise competing metrics — precision and recall""")                  
                                             
            st.write("")
            if st.checkbox("Show metric formulas", False):
                st.latex(r'''Accuracy = \frac{TN + TP}{TN + FP + TP + FN}''')   
                st.latex(r'''Precision = \frac{TP}{TP + FP}''') 
                st.latex(r'''Recall = \frac{TP}{TP + FN}''')  
                st.latex(r'''F1-score = 2 * \frac{Precision * Recall}{Precision + Recall}''')
    if genre == 'Evaluation Metrics':
        # Label encoder
        from sklearn.preprocessing import LabelEncoder
        encode = LabelEncoder()
        df[' class'] = encode.fit_transform(df[' class'])
        # Outlier
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(random_state=10,contamination=.01)
        clf.fit(df)
        y_pred_outliers = clf.predict(df)
        df['scores']=clf.decision_function(df)
        df['anomaly']=clf.predict(df.iloc[:,0:7])
        # print the anomaly
        #df[df['anomaly']==-1]
        df= df.drop(df.index[[27, 72, 192]], axis=0)
        df.reset_index(drop=True,inplace = True)
        # Split into x and y
        df = df.drop(['scores','anomaly'],axis=1)
        X = df.drop([' class'],axis=1)
        y = df[' class']
        # droping industrial_risk and operating_risk
        X.drop(['industrial_risk',' operating_risk'],axis=1,inplace=True)
        #train, test = train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
        #SVM
        # Grid Search/hyper parameter Tuning

        model = svm.SVC()
        param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,10,0.1,0.001] }]
        gsv = GridSearchCV(model, param_grid, cv=5)
        gsv.fit(X_train, y_train)
        st.write(gsv.best_params_)
        st.write("GridSearchCV Best Score = ",gsv.best_score_ )
        # Model with parameters from grid search

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        model_SVM = svm.SVC(C= 15, gamma = 0.5)
        results_SVM = cross_val_score(model_SVM, X, y, cv=kfold, scoring="accuracy")
        #st.write(results_SVM.mean())
        model_SVM.fit(X_train, y_train)
        preds = model_SVM.predict(X_test)
        f1_SVM = f1_score(y_test, preds)
        precision_SVM = precision_score(y_test, preds)
        recall_SVM = recall_score(y_test, preds)

        st.write("Training Accuracy: ", model_SVM.score(X_train, y_train))
        st.write('Testing Accuarcy: ', model_SVM.score(X_test, y_test))
        st.subheader('Classification report (using SVM)')
        st.write('F1 score is: ', f1_SVM)
        st.write('Precision is: ', precision_SVM)
        st.write('Recall is: ', recall_SVM)


    if genre == 'Predict Values':
        st.subheader("Choose only these values:")
        st.markdown("* **0.0 - Low**")
        st.markdown("* **0.5 - Medium**")
        st.markdown("* **1.0 - High**")

        def user_input_features():
            management_risk = values = st.number_input("Management Risk")
            financial_flexibility = st.number_input("Financial flexibility")
            credibility = st.number_input("Credibility")
            competitiveness = st.number_input("Competitiveness")
            data = {'management_risk':management_risk, 'financial_flexibility':financial_flexibility,'credibility':credibility,'competitiveness':competitiveness}
            features = pd.DataFrame(data,index = [0])
            return features
            
        df2 = user_input_features()

        if st.button("Go", key="Go"):
            encode = LabelEncoder()
            df[' class'] = encode.fit_transform(df[' class'])
            # Outlier
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(random_state=10,contamination=.01)
            clf.fit(df)
            y_pred_outliers = clf.predict(df)
            df['scores']=clf.decision_function(df)
            df['anomaly']=clf.predict(df.iloc[:,0:7])
            # print the anomaly
            #df[df['anomaly']==-1]
            df= df.drop(df.index[[27, 72, 192]], axis=0)
            df.reset_index(drop=True,inplace = True)
            # Split into x and y
            df = df.drop(['scores','anomaly'],axis=1)
            X = df.drop([' class'],axis=1)
            y = df[' class']
            # droping industrial_risk and operating_risk
            X.drop(['industrial_risk',' operating_risk'],axis=1,inplace=True)
            #train, test = train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
            # Building a model on SVM
            clf = svm.SVC()
            clf.fit(X_train,y_train)   
            y_pred = clf.predict(df2)
            st.header("Result :")
            st.subheader("Business goes...")
            #st.write(y_pred) 
            if y_pred==0:
                image = Image.open('bankrupt.png')
                st.image(image, width=300)
            else:
                image = Image.open('nonbankrupt1.jpg')
                st.image(image, width=300)

            



                                
    
    
