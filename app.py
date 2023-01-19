# Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns

# Disable SSL verification to allow data to be loaded from URL
# (Not necessay in all use-cases)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warning about using Matplotlib in Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load Iris dataset from URL
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Set app title and header
st.title('Exploratory Data Analysis of the Iris Dataset')
st.header('This app allows you to explore the Iris dataset and visualize the data using various plots.')

# Display DataFrame
st.subheader("DataFrame")
st.dataframe(df)

# Allow user to select a column to visualize
selected_column = st.sidebar.selectbox('Select a column to visualize', df.columns)

# Create histogram plot of selected column
st.write("Histogram Plot")
sns.histplot(df[selected_column])
st.pyplot()

# Allow user to select x and y axes for scatter plot
st.write("Scatter Plot")
x_axis = st.sidebar.selectbox('Select the x-axis', df.columns)
y_axis = st.sidebar.selectbox('Select the y-axis', df.columns)

# Create scatter plot using Plotly
fig = px.scatter(df, x=x_axis, y=y_axis)
st.plotly_chart(fig)

# Create pair plot using Seaborn
st.write("Pair Plot")
sns.pairplot(df, hue='class')
st.pyplot()

# Display description of the data
st.write("Description of the data")
st.table(df.describe())

# Display correlation matrix
st.header('Correlation Matrix')
corr = df.corr()
sns.heatmap(corr, annot=True)
st.pyplot()

# Display boxplot
st.header('Boxplot')
fig = px.box(df, y=selected_column)
st.plotly_chart(fig)

if st.sidebar.button('Show Violin Plot'):
    fig = px.violin(df, y=selected_column)
    st.plotly_chart(fig)

selected_class = st.sidebar.selectbox('Select a class to visualize', df['class'].unique())

    # if st.sidebar.button('Show Violin Plot'):
    # fig = px.violin(df[df['class'] == selected_class], y=selected_column)
    # st.plotly_chart(fig)