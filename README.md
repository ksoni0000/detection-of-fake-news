# Detection-of-Fake-News
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/ksoni0000/detection-of-fake-news)
[![Last Commit](https://img.shields.io/github/last-commit/Cartus/Automated-Fact-Checking-Literature)](https://github.com/ksoni0000/detection-of-fake-news)
[![Contribution_welcome](https://img.shields.io/badge/Contributions-welcome-blue)](https://github.com/ksoni0000/detection-of-fake-news/blob/main/contribute.md)

This data science project aims to develop a Python model capable of effectively identifying the authenticity of news articles. The approach involves constructing a TfidfVectorizer and employing a PassiveAggressiveClassifier to categorize news as either "Real" or "Fake". The project will utilize a dataset with dimensions of 7796×4 and execute all tasks within the Jupyter Lab environment.

### Prerequisites

What things you need to install the software and how to install them:

1. Python 3.6 
   - This setup requires that your machine has python 3.6 installed on it. you can refer to this url https://www.python.org/downloads/ to download python. Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are below in *how to run software section*). To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/.  
   - Setting up PATH variable is optional as you can also run program without it and more instruction are given below on this topic. 
2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/
3. You will also need to download and install below 3 packages after you install either python or anaconda from the steps above
   - Sklearn (scikit-learn)
   - numpy
   - scipy
   
  - if you have chosen to install python 3.6 then run below commands in command prompt/terminal to install these packages
   ```
   pip install -U scikit-learn
   pip install numpy
   pip install scipy
   ```
   - if you have chosen to install anaconda then run below commands in anaconda prompt to install these packages
   ```
   conda install -c scikit-learn
   conda install -c anaconda numpy
   conda install -c anaconda scipy
   ```   
   
### File Descriptions
#### Importing Libraries
##### Libraries:

numpy (imported as np): A library for numerical operations and array manipulation.
pandas (imported as pd): A library for data manipulation and analysis, used for handling datasets.
matplotlib.pyplot (imported as plt): A library for creating visualizations and plots.
seaborn (imported as sns): A library built on top of matplotlib for advanced data visualization.
warnings: A library for handling warning messages.

###### String Operations and Regular Expressions:

string: A module that provides various string-related operations and constants.
re: A module for regular expression operations, used for pattern matching and text manipulation.
Machine Learning Libraries:

sklearn.model_selection: A module that provides functions for splitting datasets into training and testing sets.
sklearn.metrics: A module that includes various metrics for evaluating machine learning models.
sklearn.feature_extraction.text: A module for extracting features from text data, specifically using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
