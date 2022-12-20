# BERT-Based Classification algorithm based on deep learning and automated labeling.
A classification of texts based on automated labeling and tranfsormers

This repository was generated and developed as part of the master thesis "Analysis and automated labeling of topics and trends in the German
automotive industry using Deep Learning and Natural Language Processing based on of industry-specific website content.".
In the context of this work, a BERT-based classification algorithm is trained and developed that can identify and classify topics and trends of a predefined industry.

## Install
[0] Install [Python](https://www.python.org/downloads/release/python-3111/) or [Anaconda](https://www.anaconda.com/products/distribution).

[1] Download Repository.
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

[3] Install requirements.
```python
pip install -r environment/requirements.txt
```

[4] Adapt and Review prefered Topics and Keywords in **TOPIC_Seed.xlsx**, **URL_Seed.xlsx**,**TOPIC_Classes.xlsx** and **CLASS_keywords.json** (only if needed!)


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

   At first three Excel Sheets are assigned: TOPIC_Seed.xlsx, URL_Seed.xlsx and TOPIC_Classes.xlsx.
   **TOPIC_Seed.xlsx** and **URL_Seed.xlsx** form the base (seed) of the websites to be crawled. The seed is crawled with help of a scrapy spider specified for the industries.(BFS)
   **TOPIC_Seed.xlsx** contains 13 Keywords per pre-defined class(german and english). Each class was derived from current literature and research.
   **URL_Seed.xlsx** contains 300 pre-selected relevant industry-specific website links (url). 
   **TOPIC_Classes.xlsx** contains 2 pre-selected relevant industry-specific website links (url) per pre-defined class (german and english). Those weblinks are later used as Centroids for K-Means Clustering.
***
#### Data cleansing and topic extraction

   The crawled websites which are based on the seed websites are cleaned in several stages.
   First, any script texts like javascript or xml are removed from the text body. Then the pre-cleaned text body is cleaned according to defined stop words. A distinction is made between industry-specific and website-specific stop words. Website specific stop words can be words like *login*, *privacy policy* or *imprint*. Industry-specific stop words can be words such as *(g/km)*, *services* or *car dealer*. After this advanced text cleaning, a language identification of the texts is performed and all texts that do not correspond to the preset language are filtered. Then the cleaned texts are lemmatized with the correct language and finally any city names are removed from the texts. 

   Next, a topic extraction is performed using the [Latent Dirichlet Allocation Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
***
#### Automated labeling
   With the help of [automated labeling](https://www.snorkel.org/features/) the cleaned data is labeled. Several labeling functions have been defined for this purpose.
   * Labeling Function 1: Keyword matching with help of pre-defined keywords per pre-defined industry-specific class, stored in **CLASS_keywords.json**
   * Labeling Function 2: Labeled data set consisting of 500 samples. The dataset was labeled by a domain expert.
   * Labeling Function 3: Trained k-Means model with fixed centroids (one centroid per predefined class). The centroids are based on the predefined classes. Two website sources (preferably wikipeda) were used per class. From these sources, the website texts are also extracted and cleaned and the topics are extracted. The topics represent the centroids for the trained k-Mean model. 

    
***
#### Classification model
   A classification model based on BERT is trained. If english is the selected language a pretrained model from tensorflow is choosen. If german is the selected language
   a pretrained model from hugging face is choosen and modified as classfier in pytorch.

## References

## Licence
This work is licensed under a [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/4.0/legalcode) and of the 
[GNU General Public License](http://www.gnu.org/licenses/).