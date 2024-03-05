import streamlit as st
import pickle
import os


DIR=DIR = 'C:\\Users\\Asus\\Desktop\\machine_learning\\iris_classifier\\'
lin_model = pickle.load(open(os.path.join(DIR+'lin_reg_model.pkl'),'rb'))
log_model = pickle.load(open(os.path.join(DIR+'log_reg_model.pkl'),'rb'))
svc_model = pickle.load(open(os.path.join(DIR+'svm_model.pkl'),'rb'))


def classify(num):
    if num<=0.5:
        return 'Setosa'
    elif num<1.5:
        return 'Versicolor'
    else:
        return 'Virginica'
    

def main():
    st.title('What is the iris?')
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white; text-align:center;">Iris Classification</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Linear Regression','Logistic Regression','SVM']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    st.spinner('Hello')
    sl = st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length',0.0,10.0)
    pw=st.slider('Select Petal Width',0.0,10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button('Classify'):
        if option=='Linear Regression':
            st.success(classify(lin_model.predict(inputs)))
        elif option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        elif option=='SVM':
            st.success(classify(svc_model.predict(inputs)))
        else:
            print("Error occur, pls try again later.")
            
if __name__=='__main__':
    main()



