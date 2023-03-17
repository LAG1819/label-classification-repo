# BERT-based text classification framework for the german industry.
A text classificatior of industry-specific texts based on automated labeling and tranfsormers.

<!-- "Analysis and automated labeling of topics and trends in the German
automotive industry using Deep Learning and Natural Language Processing based on of industry-specific website content.". -->

## Motivation
This repository was generated and developed as part of the master thesis "Development and analysis of a framework for the classification of German industry-specific texts using Deep learning and Automated Labeling for the application in the German industry using the example of the automotive industry."
In the context of this work, a Automated Labeling model and a BERT-based text classification model is trained and developed. 
In combination e.g. topics and trends of a predefined industry can be identified and classified.

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

## Usage
The generated classifier can classify random industry-specifc texts into the dedicated classes and predicts the best fitting class.
```python
example_text = "This is an example text. It contains automotive specific words like battery, electrical, loading station, autonomic driving and many more car words."
classifier = pickle.load(open(str(os.path.dirname(__file__)).split("src")[0] + r"models/classifier.pkl", 'rb')) 
predicted_topic = classifier.predict(example_text)
```
**What it offers:** A basic framwork to develop and adapt to industry specific text classification problems.  
**What it doesn't offer:** The framework is designed for user-specific requirements and must be adapted accordingly before it can be used!

## Description
It is possible to select languages german and english. The general steps to develop the classifier are as follows:

- [BERT-based text classification framework for the german industry.](#bert-based-text-classification-framework-for-the-german-industry)
  - [Motivation](#motivation)
  - [Install](#install)
  - [Usage](#usage)
  - [Description](#description)
      - [Text Mining and Crawling](#text-mining-and-crawling)
      - [Data Cleansing and Topic Extraction](#data-cleansing-and-topic-extraction)
      - [Automated Labeling](#automated-labeling)
      - [Classification Model](#classification-model)
  - [Results](#results)
    - [Automated Labeling](#automated-labeling-1)
    - [Classification](#classification)
  - [References](#references)
  - [Licence](#licence)

***
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
     

   In total there are number_classes * LF = 14 Labeling Functions generated.
 
***
#### Classification Model
   A classification model based on BERT [Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1810.04805) is trained. A classifier layer is placed on top of the pretrained models.
   The selected models are  not case-sensitive, because all cleaned texts are in lower case!
   - English Model extracted from Hugging Face: [Hugging Face team (hf-maintiners) - bert-base-uncased](https://huggingface.co/bert-base-uncased). 
   - German Model extracted from Hugging Face: [MDZ Digital Library team (dbmdz) - bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased). 

## Results
Two results each are reported for labeling and classification depending on the selected texts (Topics or Continuous texts) on which the models were trained.
### Automated Labeling
|Language|Text           | Accuracy  | Matthews Correlation Coefficent | Run |
|:------ |:-------------:|:---------:| :------------------------------:| :--:|
|DE      |TOPIC          | 0.56      | 0.48                            | 1   |
|DE      |TEXTS          | 0.47      | 0.35                            | 1   |
||||||
|EN      |TOPIC          | 0.47      | 0.37                            | 1   |
|EN      |TEXTS          | 0.40      | 0.28                            | 1   |


### Classification
|Language|Text           | Accuracy   | Matthews Correlation Coefficent  | Run |
|:------ |:-------------:|:----------:| :-------------------------------:| :--:|
|DE      |TOPIC          | 0.00       | 0.00                             | 1   |
|DE      |TEXTS          | 0.00       | 0.00                             | 1   |
||||||
|EN      |TOPIC          | 0.00       | 0.00                             | 1   |
|EN      |TEXTS          | 0.00       | 0.00                             | 1   |


## References
Programmatic Labeling: [https://proceedings.neurips.cc/paper/2016/file/6709e8d64a5f47269ed5cea9f625f7ab-Paper.pdf](https://proceedings.neurips.cc/paper/2016/file/6709e8d64a5f47269ed5cea9f625f7ab-Paper.pdf)<br>
Automated Labeling with Snorkel: [https://doi.org/10.1145/3035918.3056442](https://doi.org/10.1145/3035918.3056442)<br>
Snorkel: [https://www.snorkel.org/features/](https://www.snorkel.org/features/) <br>
<br>
Google Bidirectional Encoder Representations from Transformers: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)<br>
Hugging Face [English] Bert model:[https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)<br>
Hugging Face [German] Bert model:[https://huggingface.co/dbmdz/bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased)<br> 
## Licence
This work is licensed under a [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/4.0/legalcode) and of the 
[GNU General Public License](http://www.gnu.org/licenses/).