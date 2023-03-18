# Intro Tutorial: Topic- and Seedbased WebCrawling

In this tutorial you will find a short introduction on how to use WebCrawling with these frameworks.  
**Note: The input data used here are sample data.**

The goal is to generate a sample dataset based on the input data using the topic- and seed-based WebCrawler.
In this application, use-case specific keywords and URL must be inserted manually at first.
# Table of Contents
1. [Start a complete WebCrawling](#start-crawling)
2. [Change path to a customized seed.xlsx](#change-the-path-to-a-custom-seedxlsx-file)
3. [Topic-based WebCrawling](#topic-based-webcrawler)  
    1. [Customize Keywords](#change-keywords)
    2. [Customize number of Google search results per keyword](#change-number-google-search-results)
4. [Seed-based WebCrawler](#seed-based-webcrawler)  
    1. [Customize url](#customize-url)

### Start Crawling
* Start [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py)
  ```console
   python3 ml-classification-repo
  ```
* Select language: de or en.
* Select custom database (if requested): .feather, .xlsx or .csv.
* Execute data crawling.

### Change the path to a custom seed.xlsx file
To use a different file for crawling keywords, an alternative path can be specified. There are two ways to perform this:  
1. Open the top-level [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py) function and change the path in the main_menu(). This approach is recommended when a holistic web crawling is to be performed and the path of the seed needs to be adjusted in the process.
   ```Python3
      adjusted_path = r"my\desired\path\file.xlsx"
      crawl_data(lang, custom_seed = adjusted_path)
   ```
2. Create a TopicScraper instance and pass to it the adjusted path. This approach is recommended if only a keyword search is to be performed in a dedicated manner: 
   ```Python3
      language = 'de'
      adjusted_path = r"my\desired\path\file.xlsx"
      keyword_scraper = TopicScraper(language, s_path = adjusted_path)
      keyword_scraper.run()
   ```
*Note: The custom seed.xlsx file must have the same column names as the original framework seed file for any kind of web crawling. Otherwise it will not work.*

### Topic-based WebCrawler
#### Change Keywords:
   * Open Excel-File [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx)
   * Customize Columns:
![Alt Text](https://github.com/LGHDM/ml-classification-repo/blob/main/doc/meta/Topic_Excel.gif)
   * Note: To get a good coverage of each suspected class, one should set up at least one keyword per suspected class. The rule is: the more keywords per class, the better the data coverage achieved is likely to be.

#### Change number google search results
Let's say one wants to get only the first three Google search hits for a keyword. There are two ways to perform this:
1. Open the top-level [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py) function and change the desired number in the main_menu(). This approach is recommended when a holistic web crawling is to be performed and the number of search results needs to be adjusted in the process.
   ```Python3
      desired_number_google_search_results = 3
      crawl_data(lang, desired_number_google_search_results)
   ```
2. Create a TopicScraper instance and pass to it the desired number. This approach is recommended if only a keyword search is to be performed in a dedicated manner: 
   ```Python3
      language = 'de'
      desired_number_google_search_results = 3
      keyword_scraper = TopicScraper(language, desired_number_google_search_results)
      keyword_scraper.run()
   ```
   
### Seed-based WebCrawler
#### Customize URL:
   * Open Excel-File [**Seed.xlsx**](https://github.com/LGHDM/ml-classification-repo/blob/main/files/Seed.xlsx)
   * Customize Columns:
![Alt Text](https://github.com/LGHDM/ml-classification-repo/blob/main/doc/meta/Seed_Excel.gif)

*Note: If a selected url cannot be clearly assigned to a suspected class, GENERAL can be chosen as the class name in CLASS_K.* 

