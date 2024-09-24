---

# EDA and Deep Learning for Drug Discovery - 

## Overview

This notebook focuses on leveraging Deep learning/Machine Learning techniques for drug discovery, specifically predicting the binding affinities of small molecules to proteins. Using the **BELKA dataset** provided by Leash Biosciences, this project aims to explore a chemical space of small molecules and accelerate the identification of potential drug candidates.

The dataset contains chemical and protein information, representing molecular structure as **SMILES** strings. The primary goal is to predict whether small molecules bind to specific protein targets, assisting in the drug discovery process.

## Problem Definition

The notebook addresses a **binary classification** problem, where the target variable is `binds` (1 for binding, 0 for non-binding). The dataset includes:

- **Chemical Structures (SMILES)**: 
  - Molecule building blocks (first, second, and third).
  - Fully assembled molecules with a triazine core and DNA linker.
- **Protein Target Information**: 
  - Three protein targets (EPHX2, BRD4, ALB).
- **Binding Label (`binds`)**: 
  - A binary label indicating whether a molecule binds to a specific protein.

## Main Sections

### 1. **Data Preprocessing**
   - **Loading and Sampling**: The dataset is loaded from a Parquet file, and a sample is created for efficient exploratory analysis.
   - **Handling SMILES**: Introduction to SMILES (Simplified Molecular Input Line Entry System), a chemical notation for representing molecular structures.

### 2. **Exploratory Data Analysis (EDA)**
   - Visualizations and statistical analysis are performed on the dataset to understand patterns and distributions.
   - Examines the relationship between chemical structures and their binding to proteins.

### 3. **Modeling**
   - **Feature Engineering**: Conversion of SMILES strings into features usable by machine learning models.
   - **Model Training**: Application of MLP algorithm to predict the binding affinities.
   - **Evaluation**: The trained model is evaluated using metrics like accuracy, precision, and recall.

### 4. **Results and Discussion**
   - Summary of findings from the model.
   - Potential areas for improvement in the prediction of molecular binding.

## Dataset

The dataset used in this notebook is in **Parquet format** and contains both chemical and protein data. The SMILES strings represent molecules, while the target proteins and the binary binding label provide the supervised learning framework.

## Prerequisites

Ensure you have the following libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `rdkit` (for chemical informatics)
- `duckdb` (for data handling)

## Running the Notebook

1. **Load the dataset**: The data is stored in a Parquet file (`train.parquet`) and (`test.parquet`) on this page [Visit Kaggle](https://www.kaggle.com/competitions/leash-BELKA/code). You may modify the path based on your directory structure.
2. **Run each cell sequentially**: The notebook is designed for step-by-step execution.
3. **Model Training**: After feature extraction, models like Random Forest, Gradient Boosting, or Neural Networks can be trained.

## Future Work

- Enhance feature extraction techniques from SMILES strings.
- Experiment with different machine learning models to improve predictive performance.
- Explore larger datasets for better generalization.

## Conclusion

This project demonstrates the potential of machine learning in the drug discovery process by predicting protein-ligand binding interactions using molecular data. It highlights the importance of feature extraction from chemical notations like SMILES and provides a strong foundation for further research in computational drug discovery.

--- 

