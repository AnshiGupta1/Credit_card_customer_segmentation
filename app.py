import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# Setting up page
st.title("Customer Segmentation Analysis")

# Function to load and preprocess data
@st.cache_data  # Use the new cache_data for loading and preprocessing data
def load_data(file):
    data = pd.read_csv(file)
    st.write("### Raw Data Overview")
    st.write(data.head())
    return data

@st.cache_data  # Use the new cache_data for preprocessing
def preprocess_data(data):
    # Handle missing values for numeric columns only
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Scaling relevant numeric columns
    scaler = StandardScaler()
    columns_to_scale = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'TENURE']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    
    # Return processed data and the scaler
    return data, scaler

# Function for KMeans clustering and loading existing model
@st.cache_resource  # Use the new cache_resource for caching models
def perform_clustering(data, features, n_clusters, model_name):
    # Load the pre-trained KMeans model
    try:
        with open(f'{model_name}_kmeans_model.pkl', 'rb') as file:
            kmeans = pickle.load(file)
    except FileNotFoundError:
        # If model doesn't exist, train a new model
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(data[features])
        with open(f'{model_name}_kmeans_model.pkl', 'wb') as file:
            pickle.dump(kmeans, file)
    
    # Predict the clusters
    labels = kmeans.predict(data[features])
    data[f"Cluster_{model_name}"] = labels

    st.write(f"Model for {model_name} loaded or trained.")
    
    return labels, kmeans

# Business Scenario Visualizations
def plot_clusters(data, x, y, cluster_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, hue=cluster_column, data=data, palette='Set1', ax=ax)
    ax.set_title(f"{cluster_column} Clustering")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title="Cluster")
    st.pyplot(fig)

def plot_pair(data, features, cluster_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    pair_plot = sns.pairplot(data[features + [cluster_column]], hue=cluster_column, palette='Set1')
    st.pyplot(pair_plot)

def plot_box(data, y, cluster_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=cluster_column, y=y, data=data, ax=ax)
    ax.set_title(f"{y} by {cluster_column}")
    st.pyplot(fig)

def plot_line(data, x, y, cluster_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=cluster_column, y=y, data=data, marker='o', ax=ax)
    ax.set_title(f"{y} by {cluster_column}")
    st.pyplot(fig)

# Main Application Flow
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    raw_data = load_data(uploaded_file)
    cust_data, scaler = preprocess_data(raw_data.copy())
    
    # Scenario 1: Marketing Segmentation
    if st.button("Run Marketing Segmentation"):
        labels_1, kmeans_1 = perform_clustering(cust_data, ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT'], 3, 'Marketing')
        # Ensure that the 'Cluster_Marketing' column is present
        cust_data['Cluster_Marketing'] = labels_1
        plot_clusters(cust_data, 'BALANCE', 'PURCHASES', 'Cluster_Marketing')
    
    # Scenario 2: Comprehensive Customer Segmentation
    if st.button("Run Comprehensive Customer Segmentation"):
        labels_2, kmeans_2 = perform_clustering(cust_data, ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS'], 3, 'Comprehensive')
        # Ensure that the 'Cluster_Comprehensive' column is present
        cust_data['Cluster_Comprehensive'] = labels_2
        plot_pair(cust_data, ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS'], 'Cluster_Comprehensive')
    
    # Scenario 3: Credit Risk Analysis
    if st.button("Run Credit Risk Analysis"):
        labels_3, kmeans_3 = perform_clustering(cust_data, ['BALANCE', 'MINIMUM_PAYMENTS', 'CREDIT_LIMIT'], 3, 'Risk')
        # Ensure that the 'Cluster_Risk' column is present
        cust_data['Cluster_Risk'] = labels_3
        plot_box(cust_data, 'BALANCE', 'Cluster_Risk')
        plot_box(cust_data, 'MINIMUM_PAYMENTS', 'Cluster_Risk')
    
    # Scenario 4: Customer Behavior Analysis
    if st.button("Run Customer Behavior Analysis"):
        labels_4_spending, kmeans_spending = perform_clustering(cust_data, ['PURCHASES', 'PURCHASES_FREQUENCY'], 3, 'Spending')
        cust_data['Cluster_Spending'] = labels_4_spending
        plot_line(cust_data, 'Cluster_Spending', 'PURCHASES', 'Cluster_Spending')
        
        labels_4_retention, kmeans_retention = perform_clustering(cust_data, ['PAYMENTS', 'TENURE'], 3, 'Retention')
        cust_data['Cluster_Retention'] = labels_4_retention
        plot_box(cust_data, 'PAYMENTS', 'Cluster_Retention')

    # Show count and percentage of each cluster
    st.write("### Cluster Distribution")
    for col in cust_data.columns:
        if "Cluster" in col:
            cluster_counts = cust_data[col].value_counts()
            total_count = cluster_counts.sum()
            st.write(f"Distribution for {col}:")
            for cluster, count in cluster_counts.items():
                percent = (count / total_count) * 100
                st.write(f"Cluster {cluster}: {count} customers ({percent:.2f}%)")
