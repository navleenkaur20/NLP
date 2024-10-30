# Sentiment Classification Using Naive Bayes and Logistic Regression 
```This notebook demonstrates a sentiment classification task using Naive Bayes and Logistic Regression with two feature extraction methods: Bag of Words (BoW) and TF-IDF.```

 ## Prerequisites To run this code, 
Installed Python and Jupyter Notebook. The following Python libraries are also required:
 - `pandas` 
- `scikit-learn` 
- `nltk`

**NLTK Data Setup**
<p> To run the text preprocessing functions (e.g., lemmatization), you’ll need NLTK's wordnet and punkt data. 
Install them with: </p>

- import nltk
- nltk.download('punkt')
- nltk.download('wordnet')

## Instructions
**Open the Notebook:**

- Load the notebook file Nlp assignment 2.ipynb in Jupyter Notebook or JupyterLab.
- Run the Cells in Order:

- Each cell contains a specific step of the workflow, including data preprocessing, feature extraction, and model training.
- Run each cell sequentially to ensure the code executes correctly.
Code Structure:

### Data Preprocessing: 
Cleans the text data, tokenizes it, and performs lemmatization.
### Feature Engineering:
Converts text data into numerical features using Bag of Words and TF-IDF.
### Model Training:
Trains both Naive Bayes and Logistic Regression models using the features.
### Evaluation:
Calculates accuracy, precision, and recall for each model.
### Viewing Results:

<p> After running the evaluation cells, you’ll see the model predictions along with accuracy, precision, and recall scores.
Reproducing Results
The dataset is split into training and test sets with a set random seed (random_state=42) to ensure reproducibility. You can adjust hyperparameters and re-run cells to observe changes in model performance. </p>

