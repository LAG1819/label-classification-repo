# Tutorial 01: Topic- and Seed-based Webcrawling

In this tutorial you will find a short introduction on how to perform web crawling within this frameworks.  
**Note: The input data used here are sample data.**

The goal is to generate a sample dataset based on a customizable list of keywords/topics and url used by the topic- and seed-based WebCrawler.
The underlying processes and documentation can be found in [seed_scraper.py](https://github.com/LGHDM/ml-classification-repo/blob/main/src/topic_scrape/seed_scraper.py) and in [seeder.py](https://github.com/LGHDM/ml-classification-repo/blob/main/src/url_scrape/url_scrape/spiders/seeder.py)
The keywords/topics and url can be customized in [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx).  

# Table of Contents
1. [Start a complete WebCrawling](#start-crawling)
2. [Change path to a customized seed.xlsx](#change-the-path-to-a-custom-seedxlsx-file)
3. [Topic-based WebCrawling](#topic-based-webcrawler)  
    1. [Customize Keywords](#change-keywords)
    2. [Customize number of Google search results per keyword](#change-number-google-search-results)
4. [Seed-based WebCrawler](#seed-based-webcrawler)  
    1. [Customize url](#customize-url)


## Start complete WebCrawling
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
   2
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
    n
   ```

## Change the path to a custom seed.xlsx file
To use a different file for crawling keywords, an alternative path can be specified. There are two ways to perform this:  
#### 1. Start [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py)
* Start main in terminal and the main menu will show up.
  ```console
   python3 ml-classification-repo
  ```
* Select language: 
   ```Python3
      Please select Language.
      1: German 
      2: English
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
   2
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
      y
      Please insert absolute path (str) to custom data. Supported extensions right now are (.feather, .xlsx, .csv):
      r"my\desired\path\file.xlsx"
   ```

<!-- 2. Open the top-level [main](https://githu
b.com/LGHDM/ml-classification-repo/blob/main/__main__.py) function and change the path in the main_menu(). This approach is not generally recommended as recomandation 1.
   ```Python3
      adjusted_path = r"my\desired\path\file.xlsx"
      crawl_data(lang, custom_seed = adjusted_path)
   ``` -->
#### 2. Create a TopicScraper instance and pass to it the adjusted path. Add the seeder.crawl_data() to start the seed_crawling subsequently:
   ```Python3
      from src.topic_scrape.seed_scraper import *
      from src.url_scrape.url_scrape.spiders import seeder

      language = 'de'
      adjusted_path = r"my\desired\path\file.xlsx"

      keyword_scraper = TopicScraper(language, s_path = adjusted_path)
      keyword_scraper.run()

      seeder.crawl_data(language)
   ```
*Note: The custom seed.xlsx file must have the same column names as the original framework seed file for any kind of web crawling. Otherwise it will not work.*  


## Topic-based WebCrawler
### Change Keywords:
   * Open Excel-File [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx)
   * Customize Columns:
![Alt Text](https://github.com/LGHDM/ml-classification-repo/blob/main/doc/meta/Topic_Excel.gif)
   *Note: To get a good coverage of each suspected class, one should set up at least one keyword per suspected class. The rule is: the more keywords per class, the better the data coverage achieved is likely to be.*

### Change number google search results
Let's say one wants to get only the first three Google search hits for a keyword. There are two ways to perform this:  

#### 1. Open the top-level [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py) function and change the desired number in the main_menu().
   ```Python3
      desired_number_google_search_results = 3
      crawl_data(lang, number = desired_number_google_search_results)
   ```
#### 2. Create a TopicScraper instance and pass to it the desired number. Add the seeder.crawl_data() to start the seed_crawling subsequently:
   ```Python3
      from src.topic_scrape.seed_scraper import *
      from src.url_scrape.url_scrape.spiders import seeder

      language = 'de'
      desired_number_google_search_results = 3
      keyword_scraper = TopicScraper(language, n_results = desired_number_google_search_results)
      keyword_scraper.run()

      seeder.crawl_data(language)
   ```


## Seed-based WebCrawler
### Customize URL:
   * Open Excel-File [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx)
   * Customize Columns:
![Alt Text](https://github.com/LGHDM/ml-classification-repo/blob/main/doc/meta/Seed_Excel.gif)

*Note: If a selected url cannot be clearly assigned to a suspected class, GENERAL can be chosen as the class name in CLASS_K.* 

