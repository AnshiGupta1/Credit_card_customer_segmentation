# bank_credit_card_customers_segmentation_using_unsupervised_k_means_clustering_analysis
This project implements a Streamlit web application for segmenting credit card customers using *K-Means clustering*. The app allows users to upload a dataset, configure clustering parameters, and visualize the results interactively.

The project involves below steps in the life-cycle and implementation.  

1. Data Exploration, Analysis and Visualisations 
2. Data Cleaning 
3. Data Pre-Processing and Scaling 
4. Model Fitting 
5. Model Validation using Performance Quality Metrics namely WCSS, Elbow Method and Silhouette Coefficient/Score 
6. Optimized Model Selection with appropriate number of clusters based on the various Performance Quality Metrics 
7. Analysis Insights and Interpretations of 2 different business scenarios with various Visualisations

## Technologies Used

- *Python*
- *Streamlit*: Web app framework for data science and machine learning apps.
- *Pandas*: For data manipulation and analysis.
- *Seaborn/Matplotlib*: For data visualization.
- *Scikit-learn*: For K-Means clustering and scaling.
- *Yellowbrick*: For visualizing clustering metrics.

## Setup and Installation

1. Clone this repository:
    bash
    git clone https://github.com/AnshiGupta1/credit-card-clustering.git
    cd credit-card-clustering
    

2. Install required dependencies:
    bash
    pip install -r requirements.txt
    

    If you don't have a requirements.txt, you can manually install the dependencies:
    bash
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn yellowbrick
    

3. Run the Streamlit app:
    bash
    streamlit run app.py
    

4. Upload your customer dataset (CSV file) via the app interface and start exploring the clusters.

## Usage

1. Launch the app as described above.
2. In the sidebar, upload your CSV file containing credit card customer data.
3. Use the slider in the sidebar to choose the number of clusters (k).
4. Visualize the clustering results using various plots such as the K-Elbow and Silhouette score.
5. View the customer segmentation on a scatter plot.

## Sample Dataset

The app expects a dataset with customer features (e.g., Age, Income, SpendingScore, etc.). Ensure that the dataset does not include any non-numeric columns like CustomerID.

## Screenshots

Include screenshots of the app in action here.


## Acknowledgments

- [Streamlit](https://streamlit.io/)
