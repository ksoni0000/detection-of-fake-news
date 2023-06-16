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

##### String Operations and Regular Expressions:

string: A module that provides various string-related operations and constants.
re: A module for regular expression operations, used for pattern matching and text manipulation.
Machine Learning Libraries:

sklearn.model_selection: A module that provides functions for splitting datasets into training and testing sets.
sklearn.metrics: A module that includes various metrics for evaluating machine learning models.
sklearn.feature_extraction.text: A module for extracting features from text data, specifically using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

### Data Preprocessing

The code snippet performs text preprocessing tasks such as lowercasing the text, removing punctuation, and eliminating stopwords from the text data. These steps help in standardizing the text and removing irrelevant or commonly occurring words that may not contribute to the classification task of detecting fake news.

### Exploratory data analysis

The code snippet performs exploratory data analysis for the fake news detection project. It includes counting the labels, plotting label counts, analyzing word frequencies, and creating word cloud visualizations for both fake and real news words. These visualizations provide insights into the distribution of labels and the most frequent words in each category, aiding in understanding the dataset and potentially guiding further analysis and modeling decisions.

### Modeling and Testing the Model

Confusion matrix plotting function:
The code includes a function, plot_cm, to plot a confusion matrix.
This function takes the confusion matrix, classes, and other optional parameters.
It visualizes the confusion matrix with appropriate labels and colors.

Data splitting:
The code splits the data into the input (X) and target (y) variables.
X represents the text data, and y represents the corresponding labels (fake or real).

TF-IDF vectorization:
The TfidfVectorizer is initialized to convert text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) representation.
The vectorizer is fitted on the training data (X_train) and transformed on the test data (X_test).

Model selection and training:
The PassiveAggressiveClassifier model is chosen and initialized.
The model is trained on the TF-IDF transformed training data and the corresponding labels.

Accuracy calculation:
The model's predictions are obtained using the trained classifier on the TF-IDF transformed test data (tfidf_test).
The accuracy of the model is calculated using the accuracy_score function by comparing the predictions (y_pred) with the actual labels (y_test).

Confusion matrix visualization:
The confusion matrix is calculated using the confusion_matrix function by comparing the predictions (y_pred) with the actual labels (y_test).
The plot_cm function is called to visualize the confusion matrix.

Model testing:
The code defines a function, model_testing, that takes news text as input.
The input news text is preprocessed (converted to lowercase and cleaned) and then transformed using the same TF-IDF vectorizer.
The preprocessed news text is passed through the trained model to make predictions.
The predicted label is outputted.

### Installing and steps to run the software

A step by step series of examples that tell you have to get a development env running

1. The first step would be to clone this repo in a folder in your local machine. To do that you need to run following command in command prompt or in git bash
```
$ git clone https://github.com/ksoni0000/detection-of-fake-news.git
```

2. This will copy all the data source file, program files and model into your machine.

3.
   - If you have chosen to install anaconda then follow below instructions
     - After all the files are saved in a folder in your machine. If you chosen to install anaconda from the steps given in 	               ```Prerequisites``` sections then open the anaconda prompt, change the directory to the folder where this project is saved in     your machine and type below command and press enter.
	```
	cd C:/your cloned project folder path goes here/
	```
     - Once you are inside the directory call the ```prediction.py``` file, To do this, run below command in anaconda prompt.
	```
	python prediction.py
	```
     - After hitting the enter, program will ask for an input which will be a piece of information or a news headline that you 	    	   want to verify. Once you paste or type news headline, then press enter.

     - Once you hit the enter, program will take user input (news headline) and will be used by model to classify in one of  categories of "True" and "False". Along with classifying the news headline, model will also provide a probability of truth associated with it.

4.  If you have chosen to install python (and did not set up PATH variable for it) then follow below instructions:
    - After you clone the project in a folder in your machine. Open command prompt and change the directory to project directory by running below command.
    ```
    cd C:/your cloned project folder path goes here/
    ```
    - Locate ```python.exe``` in your machine. you can search this in window explorer search bar. 
    - Once you locate the ```python.exe``` path, you need to write whole path of it and then entire path of project folder with ```prediction.py``` at the end. For example if your ```python.exe``` is located at ```c:/Python36/python.exe``` and project folder is at ```c:/users/user_name/desktop/fake_news_detection/```, then your command to run program will be as below:
    ```
    c:/Python36/python.exe C:/users/user_name/desktop/detection-of-fake-news/prediction.py
    ```
    - After hitting the enter, program will ask for an input which will be a piece of information or a news headline that you 	    	   want to verify. Once you paste or type news headline, then press enter.

    - Once you hit the enter, program will take user input (news headline) and will be used by model to classify in one of  categories of "True" and "False". Along with classifying the news headline, model will also provide a probability of truth associated with it. It might take few seconds for model to classify the given statement so wait for it.

5.  If you have chosen to install python (and already setup PATH variable for ```python.exe```) then follow instructions:
    - Open the command prompt and change the directory to project folder as mentioned in above by running below command
    ```
    cd C:/your cloned project folder path goes here/
    ```
    - run below command
    ```
    python.exe C:/your cloned project folder path goes here/
    ```
    - After hitting the enter, program will ask for an input which will be a piece of information or a news headline that you 	    	   want to verify. Once you paste or type news headline, then press enter.

    - Once you hit the enter, program will take user input (news headline) and will be used by model to classify in one of  categories of "True" and "False". Along with classifying the news headline, model will also provide a probability of truth associated with it.
