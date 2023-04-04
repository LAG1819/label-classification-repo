# Automated Labeling Framework and BERT-based Text Classification for German Industry.
A way to quickly and automatically label data and train supervised machine learning algorithms on it, such as text classification, suitable for industry-specific cases.

## Table of contents
- [Framework for automated data labling and BERT-based text classification for the german industry.](#automated-labeling-framework-and-bert-based-text-classification-for-german-industry)
  - [Motivation](#motivation)
  - [Install](#installation)
  - [Usage](#usage)
  - [Description](#description)
      - [Text Mining](#text-mining-and-crawling)
      - [Data Processing](#data-cleansing-and-topic-extraction)
      - [Automated Labeling](#automated-labeling)
      - [Classification](#classification-model)
  - [Results](#results)
    - [Automated Labeling](#automated-labeling-1)
    - [Classification](#classification)
  - [References](#references)
  - [Licence](#licence)

## Motivation
This repository was generated and developed as part of the master thesis "Development and analysis of a framework for the classification of industry-specific texts by using Deep Learning and Automated Labeling for the application in the German industry based on the example of the automotive industry.."
In the context of this work, a label model from Snorkel is trained to achive automated labeling of unlabeled data and a BERT-based text classification model is trained and developed based on the labeled data. 
In combination, this should allow industry-specific problems from the field of Natural Language Processing to be solved in a specifiable and application-oriented manner.
It is possible to select languages german and english.

## Installation
[0] Install [Python](https://www.python.org/downloads/release/python-3111/) or [Anaconda](https://www.anaconda.com/products/distribution).

[1] Download Repository:

```python
git clone https://github.com/LGHDM/ml-classification-repo.git
```
[2.0] Start and install requirements:

> [2.1] Install requirements directly:
>```python
>pip install -r environment/requirements.txt
>```
>[2.2] Create virtual environment and install requirements on venv:

>Anaconda:
>```python
>conda env create -f environment.yml
>source activate metal
>```
>VS Code:
>```python
>py -3 -m venv .venv
>.venv\scripts\activate
>```
[3] Add Github Token in [GITHUB_TOKEN.json](https://github.com/LGHDM/ml-classification-repo/blob/main/files/GITHUB_TOKEN.json)
```console
{"GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
```
[4] Adapt and Review prefered Classes in [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx) (only if needed!):
- Seed data:
   - Labeling classes and related seed keywords to crawl: CLASS,KEYWORD_DE/KEYWORD_EN
   - Companies and others related to industry and its url: KEYWORD, URL  
- Class data for labeling:
   - Labeling classes and related url links (two per class) for Cluster Centroids: KMEANS_CLASS, KMEANS_URL, KMEANS_LANG
- Labeling classes and matching keywords:
   - Predfined labeling classes: AUTONOMOUS; ELECTRIFICATION, CONNECTIVITY, SHARED, SUSTAINABILITY, DIGITALISATION, INDIVIDUALISATION.

[5] Adapt and Review prefered Classes in **Data Cleaning, Automated Labeling and Classification** (only if needed!):
- [Data Cleaning](https://github.com/LGHDM/ml-classification-repo/tree/main/src/cleans):
   - Adapt industry-specific stopwords.
- [Automated Labeling](https://github.com/LGHDM/ml-classification-repo/tree/main/src/automated_label):
   - Adapt requested classes to assign to
   - Adapt Label Functions (LF) corresponding to requested classes.
- [Classification](https://github.com/LGHDM/ml-classification-repo/tree/main/src/classification):
   - Adapt number of classes (number labels) upon invoking the classifier model

[6] Start main menu OR customize framework:

>[6.1] Start main menu and follow the instructions:
>- Automatic start of the menu:
>```console
>ml-classification-repo
>```
>- "Manual" start of the menu:
>```python
>py __main__.py
>```
>[6.2] Customize framework:
>Please follow the tutorials in the documentations for further help and information.

## Usage
The generated label modell and classification model can label and classify random industry-specifc texts into the dedicated classes and predicts the best fitting class.
- Label model:
```python
from src.automated_label.label import *
col = 'TOPIC'
language = 'de'
example_text = "Autonomes Fahren ermöglicht es, dass Fahrzeuge selbstständig und ohne menschliches Eingreifen sicher auf den Straßen unterwegs sind."
Labeler(lang = lang, column = col).predict_label(test_text)
```

- Classification model:
```python
from src.classifier.classifier_model import predict
language = 'de'
example_text = "Autonomes Fahren ermöglicht es, dass Fahrzeuge selbstständig und ohne menschliches Eingreifen sicher auf den Straßen unterwegs sind."
predict(language,example_text)
```
**What it offers:** A basic framwork to develop and adapt to industry specific labeling problems and using supervised machine-learning like classification on top.  
**What it doesn't offer:** The framework is designed for user-specific requirements and must be adapted accordingly before it can be used!

#### Text Mining and Crawling 

   At first one Excel Sheet is assigned: **Seed.xlsx.** This file contains the following sets of data:
   - A base (seed) of website links and a list of keywords to be crawled. The seed is crawled with help of a scrapy spider specified for the industries (BFS). 
      - seed  of website links (CLASS_K, KEYWORD, URL), manually selected.
      - seed of keywords (CLASS, KEYOWRD_DE, KEYWORD_EN), manualy selected.
   - A base of website links and classes (two per class) used as Centroids for K-Means Clustering for further labeling, manually selected.
***
#### Data Cleansing and Topic Extraction

   The crawled websites are cleaned in several stages.
   1. Basic text cleaning : Any script texts (like javascript or xml) are removed from the text body. 
   2. Advanced text cleaning: Removal of stopwords. 
      - industry-specific stopwords: stopwords such as *(g/km)*, *services* or *car dealer*.
      - website-specific stopwords: stopwords like *login*, *privacy policy* or *imprint*.  
   3. Language detection of the texts and filtration of not corresponding texts to a preset language (en/de). 
   4. Lemmatization of the texts. <br>
   5. [Optional] Removal of city names in texts.

   After text cleaning a topic extraction is performed using the [Latent Dirichlet Allocation Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
***
#### Automated Labeling
   A Label Model ist trained to automize the labeling of the whole dataset. Two types of Labeling Functions were generated. 
   * Labeling Function Type 1: Keyword matching with help of pre-defined keywords. The [keywords](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx) are user-defined and case specific!
   * Labeling Function Type 2: Trained k-Means model with fixed centroids. The [K-Means Centroids](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx) are user-defined and case specific! 

   Predefined classes based on literature: 
   - AUTONOMOUS, 
   - CONNECTIVITY, 
   - DIGITALISATION, 
   - ELECTRIFICATION, 
   - INDIVIDUALISATION, 
   - SHARED, 
   - SUSTAINABILITY.<br>
     

   In total there are number_classes * 2 LF-types = 14 Labeling Functions generated.

   There are two ways to label a dataset with this framework:
   1. Total data labeling:
   Labeling Functions of type 1 and 2 are applied on the dataset. No data point in the data set is considered irrelevant.
   2. Partial data labeling:
   Labeling Functions of type 1 are applied on the dataset. Several data points in the data set might be considered as irrelevant.
 
***
#### Classification Model
   A classification model based on BERT [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) is trained on automatic labeled data. A classifier layer is placed on top of the pretrained models.
   The selected models are  not case-sensitive, because all cleaned texts are in lower case!
   - English Model extracted from Hugging Face: [Hugging Face team (hf-maintiners) - bert-base-uncased](https://huggingface.co/bert-base-uncased). 
   - German Model extracted from Hugging Face: [MDZ Digital Library team (dbmdz) - bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased). 

## Results
For the two model developments of the Automated Labeling and Classification phases, the results of the best runs are shown below for Accuracy and Matthews Correlation Coefficent (MCC). 
>A differentiation can be made between the language (German or English) and the text type (continuous text (URL_TEXT) or topics (TOPIC)) with which the models of the particular phase have been trained. 
### Automated Labeling
|Language|Text           | Accuracy  | MCC        | 
|:------ |:-------------:|:---------:| :---------:| 
|DE      |TOPIC          | 0.59      | 0.55       | 
|DE      |TEXTS          | 0.47      | 0.35       | 
|EN      |TOPIC          | 0.59      | 0.5        | 
|EN      |TEXTS          | 0.47      | 0.35       | 

### Classification
|Language|Text           | Accuracy   | MCC        | 
|:------ |:-------------:|:----------:| :---------:| 
|DE      |TOPIC          | 0.86       | 0.83       |  
|EN      |TOPIC          | 1.00       | 0.00       | 



## References
- Programmatic Labeling: [https://proceedings.neurips.cc/2016/file/Paper.pdf](https://proceedings.neurips.cc/paper/2016/file/6709e8d64a5f47269ed5cea9f625f7ab-Paper.pdf)<br>
- Automated Labeling with Snorkel - Paper: [https://doi.org/10.1145/3035918.3056442](https://doi.org/10.1145/3035918.3056442)<br>
- Snorkel Package: [https://www.snorkel.org/features/](https://www.snorkel.org/features/) <br>
- Google Bidirectional Encoder Representations from Transformers: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)<br>
- Hugging Face [English] Bert model:[https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)<br>
- Hugging Face [German] Bert model:[https://huggingface.co/dbmdz/bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased)<br> 
## License
This work is licensed under a [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/4.0/legalcode) and of the 
[GNU General Public License](http://www.gnu.org/licenses/).