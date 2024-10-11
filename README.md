# Wine Quality Prediction

This project aims to predict the quality of red wine based on various physicochemical properties. The dataset used is the Wine Quality dataset, which contains several features related to wine characteristics and a target variable representing the quality rating.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Analysis](#data-analysis)
- [Model Training](#model-training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using pip:

```bash
pip install -r requirements.txt

Usage

    Clone the repository:

    git clone https://github.com/lijoks/wine-quality-prediction.git
    cd wine-quality-prediction

    Open the Jupyter Notebook for data analysis and model training:

    jupyter notebook wine_quality_analysis.ipynb

    Run the wine_app.py file to use the trained model for predictions.

Data Analysis

The data analysis includes:

    Loading the dataset and displaying the first few records.
    Checking for missing values and data types.
    Descriptive statistics of the dataset.
    Visualizations including histograms, box plots, scatter plots, and heatmaps to understand the relationships between features.

Model Training

The project uses various supervised learning algorithms to predict wine quality. The following models are implemented:

    Random Forest Classifier
    Support Vector Classifier (SVC)
    Decision Tree Classifier

The dataset is split into training and testing sets, and the models are evaluated based on accuracy and F1-score.
Results

The results of the model training and evaluation are visualized, showing the performance of each model. The best-performing model is selected based on the evaluation metrics.
Contributing

Contributions are welcome! If you have suggestions for improvements or features, please feel free to open an issue or submit a pull request.
License

This project is licensed under the MIT License 

