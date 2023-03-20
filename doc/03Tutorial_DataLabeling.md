# Intro Tutorial: Label Data with Help of Snorkel

In this tutorial you will find a short introduction on how to train and apply a Snorkel Label Model including customized Labeling Functions.

To learn more about Snorkel visit [🚀 Snorkel.org](https://snorkel.org) or check out the [Snorkel API documentation](https://snorkel.readthedocs.io/).  

**Note: The input data used here are sample data.**  
  

**Important note: If a user-defined data set is entered, the frame work generates a training, test and validation data set from it (ratio: 60 -20 -20). The test and validation data set (marked as label_testes.xlsx and label_valset.xlsx files) are stored in the [files](https://github.com/LGHDM/ml-classification-repo/blob/main/files/03_label/). These must then be labeled manually (by a domain expert). After the labeling is done, the process can be started again and a label model is developed.**

# Table of Contents
1. [Start a complete Label Modell training.](#start-automated-labeling-and-label-modell-training)
    1. [Total data labeling](#perform-a-total-data-labeling-using-k-means-cluster-each-data-sample-gets-a-label-assigned)
    2. [Partial data labeling](#perform-a-partial-data-labeling-not-every-data-sample-gets-a-data-label)
2. [Change path to a customized dataset.](#change-dataset)
    1. [kMeans](#change-kmeans-data-for-total-data-labeling)
    2. [Automated Labeling](#change-data-to-label-for-total-and-partial-data-labeling)
3. [Customize Classes to label with.](#customize-class-labels)
4. [Add Labeling Functions.](#add-labeling-functions)
5. [Add custom hyperparameter optimization techniques](#add-hyperparameter-optimization-techniques)


# Start Automated Labeling and Label Modell training

## Perform a total data labeling using k-Means cluster (each data sample gets a label assigned)
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
* Execute data k-Means generation.
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
   5
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
    n
   ```
* Execute  automated labeling.
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
   6
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
    n
   ```
## Perform a partial data labeling (Not every data sample gets a data label)
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
* Execute automated labeling.
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
   6
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
    n
   ```


# Change Dataset
*Note: The dataset must be a .feather file*  

## Change kMeans data (for total data labeling)
There are two ways (1. and 2.) to change the dataset for kMeans and one option (3.) to change the dataset for the automated labeling. 
1. Change k-Means dataset: Change selected centroids for k-Means in [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx). Each centroid represents one class labels. Execute main after adaption.
INSERT GIF.
2. Change k-Means dataset: Loading a customized file for the zentroids of kMeans. The customized zentroids can be cleaned and topics extracted if requested.
    ```Python3
      from src.automated_label.cluster_kmeans import *

      language = 'de'
      adjusted_path_kMeans_data = r"my\desired\path\file.feather"
      
      #zentroid cleansing and topic extraction if requested
      textFilter(lang = language,adjusted_path_kMeans_data,r"files\cleaned_classes_"+lang+r".feather").run() 
      TopicExtractor(input_topic = 2,s_path = r"files\cleaned_classes_"+lang+r".feather",t_path = r"files\topiced_classes_"+lang+r".feather",lang = language,zentroid = True).run()
      
      #actual k means generation
      TOPIC_KMeans(lang = language).run()
    ```
Alternatively if custom data set containing zentroids are already cleaned: 
```Python3
from src.automated_label.cluster_kmeans import *

language = 'de'
adjusted_path_kMeans_data = r"my\desired\path\file.feather"

#actual k means generation
TOPIC_KMeans(lang = language, topics_path = adjusted_path_kMeans_data ).run()
```

## Change data to label (for total and partial data labeling)  
To use an alternative dataset for data cleansing, the same options as in Tutorial 01 and 02 can applied: Execute main and follow instructions or create a label class.   


    ```Python3
        from src.automated_label.label import *

        adjusted_path_for_label_data = r"my\other\desired\path\file.feather"
        selected_text_column = 'MY_TEXT'
        language = 'de'

        labelModel_tarining_and_application = Labeler(lang,s_path = adjusted_path_for_label_data, column = selected_text_column)
        labelModel_tarining_and_application.run()

        Out: Train, Test and Validate Dataset were generated. Please label train and validate data before further proceeding!
        Out: No labeled Test and Validate data exist! Please label generated train and test data file, stored in files/03_label/
    ```

The framework automatically generates training-, test- and validationset (ratio: 60 -20 -20) after this execution and stops the further execution with the displayed output.   
The test- and validationset must then be labeled manually (by a domain expert). After the labeling is done, the process can be started again.  

# Customize Class labels
In order to generate new or own classes for the data labeling, the class variables of an customized class can simply be set up. The customized class inherits from the Labeler class.
*Attention: The labeling functions must also be adapted accordingly!*

```Python3
from src.automated_label.label import *

class CustomLabeler(Labeler):
    CUSTOM_CLASS1 = 0
    CUSTOM_CLASS2 = 1
    .
    .
    .
```

# Add Labeling Functions
In order for new labeling functions to be applied, they must first be generated. These can be easily added to the list lfs of a Labeler instance.

```Python3
from src.automated_label.label import *

language = 'de'

def my_custom_lf():
    #do something

my_labeler = Labeler(lang = language)

#add new labeling fucntions
my_labeler.lfs += [my_custom_lf]

#overwrite and assign only new labeling functions
my_labeler.lfs = [my_custom_lf]

my_labeler.run()
```

# Add hyperparameter optimization techniques
Adding more hyperparameter optimization techniques is done quite similar as adding more labeling functions.
```Python3
from src.automated_label.label import *

language = 'de'

name_my_custom_hpo = 'custom_hpo'
def my_custom_hpo(training_set, test_set, Y_test_set, k_fold, split_i):
    #do something

# add new hyperparameter optimization techniques
my_labeler = Labeler(lang = language)
my_labeler.hpo_list += [(name_my_custom_hpo, my_custom_hpo)]

# overwrite and assign only new hyperparameter optimization techniques
my_labeler.hpo_list = [(name_my_custom_hpo, my_custom_hpo)]

my_labaler.run()

```