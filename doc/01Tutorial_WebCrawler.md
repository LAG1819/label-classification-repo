# Intro Tutorial: Topic- and Seedbased WebCrawling

In this tutorial you will find a short introduction on how to use WebCrawling with these frameworks.  
**Note: The input data used here are sample data.**

The goal is to generate a sample dataset based on the input data using the topic- and seed-based WebCrawler.
In this application, use-case specific keywords and URL must be inserted manually at first.

### Topic-based WebCrawler
#### Change Keywords:
   * Open Excel-File [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx)
   * Customize Columns:
![Alt Text](https://github.com/LGHDM/ml-classification-repo/blob/main/doc/meta/Topic_Excel.gif)
 


#### Change number google search results
   
### Seed-based WebCrawler
#### Customize URL:
   * Open Excel-File [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx)
   * Customize Columns:
![Alt Text](https://github.com/LGHDM/ml-classification-repo/blob/main/doc/meta/Seed_Excel.gif)

### Start Crawling
* Start [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py)
  ```console
   python3 ml-classification-repo
  ```
* Execute Process Step *(2) Execute data crawling.*