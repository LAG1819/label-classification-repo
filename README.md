# BERT-Based Classification algorithm based on deep learning and automated labeling.
A classification of texts based on automated labeling and tranfsormers

This repository was generated and developed as part of the master thesis "Analysis and automated labeling of topics and trends in the German
automotive industry using Deep Learning and Natural Language Processing based on of industry-specific website content.".
In the context of this work, a BERT-based classification algorithm is trained and developed that can identify and classify topics and trends of a predefined industry.

## Motivation

## Install
[0] Install [Python](https://www.python.org/downloads/release/python-3111/) or [Anaconda](https://www.anaconda.com/products/distribution).

[1] Download Repository:
```python
git clone https://github.com/LGHDM/ml-classification-repo.git
```

[2] Create virtual environment:

Anaconda:
```python
conda env create -f environment.yml
source activate metal
```
Windows:
```python
py -3 -m venv .venv
.venv\scripts\activate
```

[3] Install requirements:
```python
pip install -r environment/requirements.txt
```

[4] Adapt and Review prefered Topics and Keywords in **Seed.xlsx**(only if needed!):
- Seed data:
   - Labeling classes and related seed keywords to crawl: CLASS,KEYWORD_DE/KEYWORD_EN
   - Companies and others related to industry and its url: KEYWORD, URL  
- Class data for labeling:
   - Labeling classes and related url links (two per class) for Cluster Centroids: **TOPIC_Classes.xlsx** 
   - Labeling classes and matching keywords: **CLASS_keywords.json** 
   

## Usage
The generated classifier can classify random texts into the industry-specific topics and trends and returns the most appropriate topic to which the text is related. 
```python
example_text = "This is an example text. It contains automotive specific words like battery, electrical, loading station, autonomic driving and many more car words."
classifier = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/classifier.pkl", 'rb')) 
predicted_topic = classifier.predict(example_text)
```

## Description
It can be selected between German and English with a resulting german or english specified classifier.
The general steps to develop the classifier are as follows:

   1. [Data mining and crawling of website text content using seed url list and seed keywords.](#data-mining-and-crawling-of-website-text-content)
   2. [Data cleansing and topic extraction of website text content.](#data-cleansing-and-topic-extraction)
   3. [Automated labeling of cleaned website text content and of topics.](#automated-labeling)
   4. [Train Classification model (BERT-based).](#classification-model)

***
#### Data Mining and crawling of website text content

   At first one Excel Sheet is assigned: **Seed.xlsx.** This file contains the following sets of data:
   - A base (seed) of website links and a list of keywords to be crawled. The seed is crawled with help of a scrapy spider specified for the industries (BFS). 
      - seed  of website links (CLASS_K, KEYWORD, URL), manually selected.
      - seed of keywords (CLASS, KEYOWRD_DE, KEYWORD_EN), manualy selected.
   - A base of website links and classes (two per class) used as Centroids for K-Means Clustering for further labeling, manually selected.
***
#### Data cleansing and topic extraction

   The crawled websites are cleaned in several stages.
   1. Basic text cleaning : Any script texts (like javascript or xml) are removed from the text body. 
   2. Advanced text cleaning: Removal of stopwords. 
      - industry-specific stopwords: stopwords such as *(g/km)*, *services* or *car dealer*.
      - website-specific stopwords: stopwords like *login*, *privacy policy* or *imprint*.  
   3. Language detection of the texts and filtration of not corresponding texts to a preset language (en/de). 
   4. Lemmatization of the texts. 
   5. Removal of city names in texts. 

   After text cleaning a topic extraction is performed using the [Latent Dirichlet Allocation Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
***
#### Automated labeling
   With the help of [automated labeling](https://www.snorkel.org/features/) the cleaned data is labeled. 
   * Labeling Function 1: Keyword matching with help of pre-defined keywords (**CLASS_keywords.json**)
   * Labeling Function 2: Labeled dataset by a domain expert consisting 500 samples.
   * Labeling Function 3: Trained k-Means model with fixed centroids (according to centroids from **Seed.xlsx.**). 
   Predefined classes based on literature: AUTONOMOUS, CONNECTIVITY, DIGITALISATION, ELECTRIFICATION, INDIVIDUALISATION, SHARED, SUSTAINABILITY.
 
***
#### Classification model
   A classification model based on BERT [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) is trained. 
   - English Model based on TensorFlow trained model by [google](https://tfhub.dev/google/collections/bert/1). 
   - German Model based on huggingface trained model by [deepset](https://huggingface.co/bert-base-german-cased). A classifier layer is placed on top of pretrained model.

## References

## Licence
This work is licensed under a [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/4.0/legalcode) and of the 
[GNU General Public License](http://www.gnu.org/licenses/).