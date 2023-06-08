# Disaster Response Pipeline Project

A WEB app deployed to [Heroku](https://dashboard.heroku.com/) and accessible at the following URL:

<div align="center">
  <h3>https://udacity-disasterapp.herokuapp.com</h3>
</div>

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#file_descriptions)
4. [Instructions](#instructions)
5. [Screenshots](#screen_shots)
6. [Effect of Imbalance](#effect_imbalance)
7. [Deployeme to Heroku](#deployment_heroku)
8. [Acknowledgements](#acknowledgements)

## Project Motivation <a name="motivation"></a>

This project is part of Udacity's Data Science Nanodegree Program in collaboration with [Appen](https://appen.com/) (formally Figure 8).
The goal of this project is to build a model for an API that classifies messages received in real time during a disaster event, so that messages can be sent to the correct disaster response agency.

A dataset of thousands of real-world messages during disaster events is used to build this model.

The Project is divided in the following sections:

1. Building an ETL pipeline to extract, clean and load the data to a SQLite database.
2. Building a ML pipeline for multilabel classification.
3. Run a WEB application to get classification results for a new message and also display visualizations of the data.
4. Deploy the WEB application to [Heroku](https://dashboard.heroku.com/).

## Installation <a name="installation"></a>

This project requires Python 3 and the following Python libraries installed:

1. ML libraries: `NumPy`, `Pandas`, `SciPy`, `scikit-learn`
2. NLP libraries: `NLTK`
3. SQLlite library: `SQLalchemy`
4. Model loading and saving library: `Pickle`
5. Web app and visualization: `Flask`, `Plotly`
6. Other libraries: `Wordcloud`, `colorlover` ,`gunicorn`

`The full list of requirements can be found in requirements.txt file.`

## File Descriptions <a name="file_descriptions"></a>

- **App** folder: contains our responsive Flask Web App.
  - `run.py`: main file to run the web application.
  - `plotly_figures.py`: Returns `Plotly` figure configuration (data and layout).
  - `templates` folder: contains four html pages (`dashboard.html`, `classifier.html`, `go.html` and `model_metrics.html`).
  - `static` folder: contains our customized `CSS` file and `Bootstrap` (compiled and minified `CSS` bundles and `JS` plugins).
- **Models** folder: contains our ML pipeline.

  - `train_classifier.py`: uses grid search and AdaBoostClassifier to classify messages into 36 categories.

- **Data** folder: contains our ETL pipeline.
  - `process_data.py`: A script to build an ETL pipeline that loads the `messages` and `categories` datasets, merge, clean and save data to a SQLite database.
  - `disaster_categories.csv` and `disaster_messages.csv`: real-world datasets provided by [Appen](https://appen.com/).
- **Notebooks** folder: contains the project's notebooks.

- **Heroku_deployment** folder: used to [deploy](#deployment_heroku) the app to `Heroku`.

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database

     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

   - To run ML pipeline that trains classifier and saves to pkl

     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

   `python run.py`

3. Go to http://0.0.0.0:3001/

## Screenshots <a name="screen_shots"></a>

1. The `Message Analyser` page displays the categories into which the message has been classified, highlighted in blue.

   ![image Classifier](https://github.com/AlaGrine/udacity-disaster-response-project/blob/main/screenshots/Classifier.png)

2. THe `Dashboard` page shows the distribution of message genres, categories and lengths. A Wordcloud of the most common words is also displayed.

   ![image Dashboard](https://github.com/AlaGrine/udacity-disaster-response-project/blob/main/screenshots/Dashboard.png)

3. The `Model metrics` page displays F1-score and recall metrics of our model.

   ![image Metrics](https://github.com/AlaGrine/udacity-disaster-response-project/blob/main/screenshots/Metrics.png)

## Effect of Imbalance: <a name="effect_imbalance"></a>

### Intra-category imbalance:

There is an imbalance within each category as shown in the next chart.

Using `Random Forest` classifier as model, we can observe:

- For **10** categories, the proportion of samples in the positive class is less than **2%**, and for **20** categories, the ratio is less than **5%** .
  These categories have very low f1 scores and recall.
- The precision is around 0.7 for 29 categories and zero for 7 categories.
- The F1-score tends to increase as the proportion of samples in the positive class increases.

![image imbalance_f1score](https://github.com/AlaGrine/udacity-disaster-response-project/blob/main/screenshots/Inbalance.png)

```
F1-score scoring method is more appropriate for GridSearchCV than accuracy.
```

### Imbalance between categories:

There is also an imbalance between the 36 categories. While the Related and Air_related categories have more than 15,000 messages, other categories such as Fire or Offers have less than 300 messages.

To deal with this imbalanced dataset, we tried to upsample the minority labels using the [MLSMOTE](https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote) implementation. This implementation can be found on the notebook.

The figure below shows the effect of MLSMOTE on the f1-score of the positive class using the Random Forest classifier.

<div align="center">
  <img src="https://github.com/AlaGrine/udacity-disaster-response-project/blob/main/screenshots/MLSMOTE_effect.png" >
</div>

<br>

> - However, the `AdaBoostClassifier` algorithm still outperforms `Random Forest` even with data augmentation.
> - The pkl file of `AdaBoostClassifier` is much smaller than that of `Random Forest`.
> - Finally, `AdaBoostClassifier` is faster at classifying new messages.

For all these reasons, we decided to implement `AdaBoostClassifier` for our WEB application.

## Deployment online to Heroku: <a name="deployment_heroku"></a>

Here is the URL of the app:

### https://udacity-disasterapp.herokuapp.com/

It is deployed to [Heroku](https://dashboard.heroku.com/) using the `Heroku_deployment` folder, which contains:

- `requirements.txt` : contains the full list of requirements that will be used by Heroku.
- `nltk.txt`: tells Heroku what to install from nltk.
- `Procfile`: tells Heroku what to do when starting the app.
- `myapp.py`: imports app and is called by the Procfile.
- The remaining folders (myapp, data and models) have the same architecture and functionality as our initial code.

To run the app locally, you will need to uncomment the last line of `myapp.py`

```
# uncomment to run locally
# app.run(host='0.0.0.0', port=3001, debug=False)
```

And run the following command `python myapp.py`.

To deploy the app to Heroku, you can run the following commands in the `Heroku_deployment` directory:

```
heroku login -i
git add .
git commit -m "your commit message"
heroku create your-app-name --buildpack heroku/python
git push heroku master

```

## Acknowledgements <a name="acknowledgements"></a>

Must give credit to [Appen](https://appen.com/) for the data, and [udacity](https://www.udacity.com/) for this program.
