# BERT-Based Classification algorithm with help of automated labeling.
A classification of texts based on automated labeling and tranfsormers

This repository was developed as part of the master thesis "Analysis and automated labeling of topics and trends in the German
automotive industry using Deep Learning and Natural Language Processing based on of industry-specific website content." generated and developed. 
In the context of this work, a BERT-based classification algorithm is trained and developed that can identify and classify topics and trends of a predefined industry.

## Install

## Usage
The generated classifier can classify random texts into the industry-specific topics and trends and returns the most appropriate topic to which the text is related. 

## Description
The general steps to develop the classifier are as follows:
~~~
1. Data mining and crawling of website text content using seed url list and seed keywords.
2. Data cleansing and topic extraction of website text content (one sample is one website link)
3. Automated labeling of cleaned website text content and of topics.
4. Train Classification model (BERT-based).
~~~
***
1. **Data Mining and crawling of website text content.**
   At first three Excel Sheets are assigned: TOPIC_Seed.xlsx, URL_Seed.xlsx and TOPIC_Classes.xlsx.
   **TOPIC_Seed.xlsx** and **URL_Seed.xlsx** form the base (seed) of the websites to be crawled. The seed is crawled with help of a scrapy spider specified for the industries.(BFS)
   **TOPIC_Seed.xlsx** contains 13 Keywords per pre-defined class(german and english). Each class was derived from current literature and research.
   **URL_Seed.xlsx** contains 300 pre-selected relevant industry-specific website links (url). 
   **TOPIC_Classes.xlsx** contains 2 pre-selected relevant industry-specific website links (url) per pre-defined class (german and english). Those weblinks are later used as Centroids for K-Means Clustering.
***
2.**Data cleansing and topic extraction.**
***
3.**Automated labeling.**
**
4.**Train Classification model.(BERT-based).**

## Licence
This work is licensed under a [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/4.0/legalcode) and of the 
[GNU General Public License](http://www.gnu.org/licenses/).