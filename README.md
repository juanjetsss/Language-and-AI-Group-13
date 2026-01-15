# Language-and-AI-Group-13
This project preprocesses a Reddit posts dataset (gender.csv) and then trains/evaluates two logistic regression models (baseline vs punctuation version). To run everything on another computer, you need Python (recommended: Python 3.12.1) and the dataset file gender.csv in the same folder as the code. The dataset must contain the columns auhtor_ID, post, and female (note that the column name is spelled auhtor_ID in the code).
Before running the code, install the required Python packages. Please write in the terminal
1.	pip install pandas numpy scikit-learn spacy tqdm langdetect matplotlib joblib 
2.	py -m spacy download en_core_web_sm
