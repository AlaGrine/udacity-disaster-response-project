import sys
import pandas as pd 
import numpy as np
import re
from sqlalchemy import create_engine, text as sql_text

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
import pickle

import datetime

import warnings
warnings.simplefilter("ignore", UserWarning)

def load_data(database_filepath):
    """
    Load data from table `Disaster`.

    Parameters
    -----------
        database_filepath - SQLite file storing clean dataset (`Disaster` table), created by `process_Data.py`.
    
    Returns
    -----------
        X - Input data (message sent).
        Y - 36 categories to predict. 
        category_names : name of the categories.
    """
    # Connect to SQLite table
    connection = create_engine('sqlite:///' + database_filepath)
    query = "select * from Disaster"
    df = pd.read_sql_query(con=connection.connect(),  sql=sql_text(query))

    # X: the message column.
    X = df['message']

    # Y: the 36 categories which we need to predict.
    Y = df[df.columns[4:]] # for caterories start from column 4.

    # category_names
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and preprocess text:
        1. find urls and replace them with 'urlplaceholder'.
        2. Normalization of the text : Convert to lowercase.
        3. Normalization of the text : Remove punctuation characters.
        4. Split text into words using NLTK.
        5. remove stop words.
        6. Lemmatization.    

    Parameters
    -----------
        text: text

    Returns
    -----------
        clean_tokens
    """
    # 1. find urls and replace them with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, 'urlplaceholder', text)

    # 2. Convert to lowercase
    text = text.lower().strip() 
    
    # 3. Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # 4. Split text into words using NLTK
    words = word_tokenize(text)
    
    # 5. Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    # 6.1 Reduce words to their root form
    lemmed = [lemmatizer.lemmatize(w) for w in words]
    # 6.2 Lemmatize verbs by specifying pos
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return clean_tokens

def print_classification_report(Y_test, Y_pred,category_names):
    """
    Print a text report showing the main classification metrics, using `classification_report` method.
    Metrics: F1 score,precision and recall per category.
    
    Parameters
    ----------- 
        Y_test : actual y values. 
        Y_pred : predicted y values.
        category_names : category names.
    
    Returns
    ----------- 
        None
    """
    for indice in range(len(category_names)):
        print("Classification report for {}".format(category_names[indice]))
        print(classification_report(Y_test.iloc[:,indice], Y_pred[:,indice]))
        
def build_metrics_DF(y_test, y_pred,model_name):
    """
    Build a dataframe with the main classification metrics.
    Metrics: F1 score,precision per category (fire, water...) and class (0, 1 and weigted avg).
    
    Parameters
    ----------- 
        y_test : actual y values.
        y_pred : predicted y values.
        model_name (str) : model name.
        
    Returns
    -----------
        df_classification_report (DataFrame): A dataframe containing classification metrics 
                            per category (fire, water...) and class (0, 1 and weigted avg).
    """
    
    class_list = ["0","1","weighted avg"]
    reports_list = []
    
    for indice,category in enumerate(y_test.keys()):
        # 1. calssification report
        report = classification_report(y_test.iloc[:,indice], y_pred[:,indice], output_dict=True)
        # 2. accuracy
        try:
            accuracy = accuracy_score(y_test.iloc[:,indice], y_pred[:,indice])
        except:
            accuracy = np.nan
        # 3. roc_auc_score
        try:
            roc = roc_auc_score(y_test.iloc[:,indice], y_pred[:,indice])
        except:
            roc = np.nan
            
        # 4. Build list of dicts (will be used to create a dataframe)
        for class_name in class_list:
            try:
                report_class = report.get(class_name)
                
                # Accuracy and roc_auc scores are available only for weighted avg
                if class_name == "weighted avg":
                    report_class["accuracy"] = accuracy
                    report_class["roc"] = roc
                else:
                    report_class["accuracy"] = np.nan
                    report_class["roc"] = np.nan
                
                report_class["category"] = category 
                report_class['class'] = class_name 
                report_class['model_name'] = model_name                
                reports_list.append(report_class)
            except:
                pass
        
    # 5. create the dataframe:
    df_classification_report  = pd.DataFrame(reports_list)   
    return df_classification_report      


def build_model():
    """
    Build a ML Pipeline and grid search to optimize our machine learning workflow.
    
    Returns
    -----------  
        model - An optimized `AdaBoostClassifier` model.   
    """

    # 1.pipeline
    clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42))
    
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(clf))
                        ]) 

    # 2.Parameters:
    # Only limited parameters as training is too time consuming.
    parameters = {'clf__estimator__learning_rate':[0.5,0.8,1],
                  'clf__estimator__estimator__max_depth' : [1,2]
                  }

    # 3.GridSearchCV
    print(f"Start GridSearchCV ... {datetime.datetime.now()}")
    model = GridSearchCV(estimator = pipeline, param_grid = parameters,cv=2,verbose=10)

    return model


def evaluate_model(model, X_test, Y_test,category_names):
    """
    Print classification metrics (F1 score,precision and recall) per category.
    Save model metrics into pandas dataframe.

    Parameters
    ----------- 
        model : the classifier model.
        X_test : input from test subset. 
        Y_test : output from test subset.
        category_names : category names.
        tfidf : the TFIDF vectorizer that will be used to transform X_test.

    Output
    ----------- 
        "metrics.csv"(file) : a csv file containing model metrics.
    """

    print(f"Start Evaluate model ... {datetime.datetime.now()}")

    # 1. prediction
    Y_pred = model.predict(X_test)

    # 2. print classification report
    print_classification_report(Y_test, Y_pred,category_names)

    # 3. Build metrics dataframe using `build_metrics_DF` method, and save them to csv file.
    df_metrics = build_metrics_DF(Y_test,Y_pred,"adaBoost_cv")
    df_metrics.to_csv("./models/metrics.csv",index=False)

    
def save_model(model, model_filepath):
    """
    Save our model as a pickle file.
    """
    # 1. open a file, where we will store data
    file = open(model_filepath, 'wb')

    # 2. pickle dump
    pickle.dump(model, file)

    # 3. close the file
    file.close()


def main():
    """
    Main function to train our classifier:
        1. Load data from SQLite database
        2. Train our model on training set.
        3. Evaluate our model on test set.
        4. Save model and TFIDF Vectorizer as Pickle.
    
    """
    if len(sys.argv) == 3:

        print(f"Start  ... {datetime.datetime.now()}")

        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print(f"End  ... {datetime.datetime.now()}")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()