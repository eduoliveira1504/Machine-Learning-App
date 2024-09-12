import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('🤖 Machine Learning App')

st.info('This app builds a machine learning model!')

with st.expander('Data 🐧'):
    st.write('**Raw Data**')
    df = pd.read_excel('Data/penguins_cleaned.xlsx')
    df

    st.write('**X**')
    X_raw = df.drop('species',axis=1)
    X_raw

    st.write('**Y**')
    y_raw = df.species
    y_raw

with st.expander('Data visualization'):
    st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

with st.sidebar:
    st.header('Input features')
    island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
    bill_length_mm = st.slider('Bill Length (mm)',32.1,59.6,43.9)
    bill_depth_mm = st.slider('Bill depth (mm)',13.1,21.5,17.2)
    flipper_lenght_mm = st.slider('Flipper Length (mm)',172.0,231.0,201,0)
    body_mass_g = st.slider('Body mass (g)',2700.0,6300.0,4207.0)
    gender = st.selectbox('Gender',('Male','Female'))

# Criando um DataFrame para os recursos de "input"
data = {'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_lenght_mm': flipper_lenght_mm,
        'body_mass_g': body_mass_g,
        'sex': gender}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('**Input Features**'):
    st.write("**Input penguin**")
    input_df
    st.write('**Combined Penguins Data**')
    input_penguins


# preparando os dados
## codificando o X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix = encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# codificando o y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y

# CONSTRUINDO O MODELO & INFERENCIA
## treinando o modelo de machine learning
clf = RandomForestClassifier()
clf.fit(X,y)

## aplicando o modelo para predições
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie','Chinstrap','Gentoo']
df_prediction_proba.rename(column = {0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# exibição das espécies prevista:
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config = {
                'Adelie': st.column_config.ProgressColumn(
                   'Adelie',
                   format = '%f',
                   width = 'medium',
                   min_value = 0,
                   max_value = 1
                ),
                'Chinstrap': st.column_config.ProgressColumn(
                   'Chinstrap',
                   format = '%f',
                   width ='medium',
                   min_value = 0,
                   max_value = 1
                ),
                'Gentoo': st.column_config.ProgressColumn(
                   'Gentoo',
                   format = '%f',
                   width = 'medium',
                   min_value = 0,
                   max_value = 1
                )
             }, hide_index = True)

df_prediction_proba

penguins_species = np.array(['Adeline','Chinstrap','Gentoo'])
st.success(str(penguins_species[prediction[0]]))

