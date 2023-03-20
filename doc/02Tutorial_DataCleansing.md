# Tutorial 02: DataCleansing and how to add additional cleansing

In this tutorial it will be shown how a text cleanup of texts can be done using the framework presented here.
**Note: The input data used here are sample data.**

The goal is to achieve a cleanup of the input texts that is as comprehensive as possible.
The underlying processes and documentation can be found in [clean.py](https://github.com/LGHDM/ml-classification-repo/blob/main/src/cleans/clean.py) and [topic_lda.py](https://github.com/LGHDM/ml-classification-repo/blob/main/src/cleans/topic_lda.py). 

# Table of Contents
1. [Data Cleansing](#data-cleansing)    
    1. [Start a complete DataCleansing](#start-data-cleansing)
    2. [Change path to a customized dataset](#change-dataset)
    3. [Customize Stopwords](#customize-stopwords)
    4. [Add Function for data cleansing](#add-data-cleansing-function)
2. [Topic Extraction](#topic-Extraction)

# Data Cleansing 
## Start Data Cleansing
* Start [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py) in terminal and the main menu will show up.
  ```console
   python3 ml-classification-repo
  ```
* Select language: 
   ```Python3
      Please select Language.
      1: German 
      2: English
   1
   ```
* Execute data crawling.
   ```Python3
      Please select an Option:
      (0) End session.
      (1) Execute full process.
      (2) Execute data crawling.
      (3) Execute data cleaning.
      (4) Execute topic extraction.
      (5) Execute automated labeling: k-Means. 
      (6) Execute automated labeling: Train and Apply Label Modell.
      (7) Execute classification: Train and Apply Classification Modell.
   3
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
    n
   ```

## Change Dataset
#### 1. Start [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py) in terminal and the main menu will show up.
  ```console
   python3 ml-classification-repo
  ```
* Select language: 
   ```Python3
      Please select Language.
      1: German 
      2: English
   1
   ```
* Execute data crawling.
   ```Python3
      Please select an Option:
      (0) End session.
      (1) Execute full process.
      (2) Execute data crawling.
      (3) Execute data cleaning.
      (4) Execute topic extraction.
      (5) Execute automated labeling: k-Means. 
      (6) Execute automated labeling: Train and Apply Label Modell.
      (7) Execute classification: Train and Apply Classification Modell.
   3
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
      y
      Please insert absolute path (str) to custom data. Supported extensions right now are (.feather, .xlsx, .csv):
      r"my\desired\path\file.xlsx"
   ```
#### 2. Create a TextFilter Class and change path
     ```Python3
      from src.cleans.clean import *

      language = 'de'
      adjusted_path = r"my\desired\path\file.xlsx"

      text_filter = textFilter(language,s_path = adjusted_path)
      text_filter.run()

   ```
## Change chunking of dataset for DataCleansing
The given dataset is split by default into chunks of size 300 sample during data cleansing to achieve a higher execution rate and to achieve a low loss rate in case of process interruptions. The chunk size can be adjusted.

    ```Python3
      from src.cleans.clean import *

      language = 'de'
      desired_chunk_size = 1200

      text_filter = textFilter(language, chunk_size = desired_chunk_size)
      text_filter.run()
   ```

## Customize Stopwords
#### Create a TextFilter Class and add custom stopwords list
    ```Python3
      from src.cleans.clean import *

      language = 'de'
      my_stopwords_list = ["stopword1","stopword2"]

      text_filter = textFilter(language,stopwords = my_stopwords_list)
      text_filter.run()
   ```

## Customize Pattern
#### Create a TextFilter Class and add custom pattern list
    ```Python3
      from src.cleans.clean import *

      language = 'de'
      my_pattern_list = ["rgx41","njpk"]

      text_filter = textFilter(language,pattern = my_pattern_list)
      text_filter.run()
   ```

## Add Data Cleansing Function or Change Data Cleansing
Currently the only way to change the data cleansing or to add another function for data cleansing is possible by overwriting the run-function of the textFilter class.

```Python3
      from src.cleans.clean import *
      
      class textFilterCustom(textFilter):
            def custom_cleaning_function(self):
                #do something

            def run():
                try:
                    **self.custom_cleaning_function()**
                except Exception as e:
                    print(e)
                    return

      language = 'de'
      text_filter = textFilter(language)
      text_filter.run()
   ```


# Topic extraction

## Start Topic Extraction
* Start [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py) in terminal and the main menu will show up.
  ```console
   python3 ml-classification-repo
  ```
* Select language: 
   ```Python3
      Please select Language.
      1: German 
      2: English
   1
   ```
* Execute data crawling.
   ```Python3
      Please select an Option:
      (0) End session.
      (1) Execute full process.
      (2) Execute data crawling.
      (3) Execute data cleaning.
      (4) Execute topic extraction.
      (5) Execute automated labeling: k-Means. 
      (6) Execute automated labeling: Train and Apply Label Modell.
      (7) Execute classification: Train and Apply Classification Modell.
   4
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
    n
   ```

## Change input dataset
 ```console
   python3 ml-classification-repo
  ```
* Select language: 
   ```Python3
      Please select Language.
      1: German 
      2: English
   1
   ```
* Execute data crawling.
   ```Python3
      Please select an Option:
      (0) End session.
      (1) Execute full process.
      (2) Execute data crawling.
      (3) Execute data cleaning.
      (4) Execute topic extraction.
      (5) Execute automated labeling: k-Means. 
      (6) Execute automated labeling: Train and Apply Label Modell.
      (7) Execute classification: Train and Apply Classification Modell.
   4
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
      y
      Please insert absolute path (str) to custom data. Supported extensions right now are (.feather, .xlsx, .csv):
      r"my\desired\path\file.xlsx"
   ```

## Change number of topics
```Python3
      from src.cleans.topic_lda import *

      language = 'de'
      desired_nbr_topics = 3

      topic_extractor = TopicExtractor(lang = language, input_topic = desired_nbr_topics).run()

   ```
