from bs4 import BeautifulSoup
from urllib.parse import urlencode, urlunparse
from urllib.request import urlopen, Request
import requests
import re
import time
import pandas as pd
import lxml.html.clean as clean
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import os
import pandas as pd
import logging
import secrets_git_luisa as s

os.environ['GH_TOKEN'] = s.github_token
os.environ['WDM_LOG'] = str(logging.NOTSET)
os.environ['WDM_LOCAL'] = '1'
os.environ['WDM_SSL_VERIFY'] = '0'

# Webcrawler - crawls for website and service information
# Saves information in a csv. No specific Return Value.
# :param searchlist: a list of strings containing seller+city per item
# :type searchlist: list
class TopicScraper:
    """Performs a Google search of all keywords defined in TOPIC_Seed.xlsx, retrieves the top 10 result links. 
    The result links will be unionend with all links defined in URL_Seed.xlsx and saves them in a new file Seed.feather.
    Seed.feather is 

    Args:
        CrawlSpider (_type_): _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    def __init__(self):
        """Initialisation of Crawler. Sets Selenium Browser and calls load_data.
        """
        self.headers = {'Accept-Langugage':'de;q=0.7',
                   'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Edge/100.01185.39"}

        self.package_dir = str(os.path.dirname(__file__)).split("src")[0]
        self.topics_df, self.url_df = self.load_data()

        self.options = webdriver.FirefoxOptions()
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--headless')
        self.options.add_argument(f'user-agent={self.headers}')
        self.browser = webdriver.Firefox(service=Service(GeckoDriverManager().install()),options=self.options)
        self.wait = WebDriverWait(self.browser, 20)

    def load_data(self):
        """Call of both Seed files. TOPIC_Seed.xlsx contains keywords to crawl. URL_Seed.xlsx contains links to crawl.

        Returns:
            DataFrame: Return loaded predefined topic(keywords) and url links each as pandas DataFrame.  
        """
        absolute_path = os.path.join(self.package_dir,r'files\TOPIC_Seed.xlsx')
        absolute_path_url = os.path.join(self.package_dir,r'files\URL_Seed.xlsx')

        topic_data = pd.read_excel(absolute_path,header = 0) 
        url_data = pd.read_excel(absolute_path_url,header = 0) 
  
        return topic_data,url_data
        
    def get_url(self,q):
        try:
            resultLinks = self.google_search(q)
        except Exception as e:
            resultLinks =[]
            print("BS4 Google Search went wrong: ",e)

        if not resultLinks:
            try:
                resultLinks = self.google_search_selenium(q)
            except Exception as e:
                resultLinks =[]
                print("SELENIUM Google Search went wrong: ",e)
                        
        return resultLinks

    def google_search(self,query):
        params = {"q": query}
        req = requests.get('https://www.google.com/search?q=',headers = self.headers, params = params,verify= False).text
        
        soup = BeautifulSoup(req,"html.parser")
        searchResults = soup.find_all(class_="g")
       

        return list(filter(self.filter_resultLinks,[sr.find("a")["href"] for sr in searchResults]))

    def google_search_selenium(self,query):
        ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
        self.browser.get('http://www.google.com')
        try: 
            frame = self.browser.find_element(By.XPATH,'//*[@id="cnsw"]/iframe') #<-locating chrome cookies consent frame 
            self.browser.switch_to.frame(frame) 
            self.browser.find_element(By.XPATH,'//*[@id="introAgreeButton"]').click()#<-looking for introAgreeButton button, but seems google has changed its name since and  it only works in old chrome versions.

        except NoSuchElementException:
            self.browser.find_element(By.XPATH,'//*[@id="L2AGLb"]').click() #<- pay attention to new id.

        search = self.browser.find_element(By.NAME,'q')
        search.send_keys(str(query))
        search.send_keys(Keys.RETURN) 

        wait = WebDriverWait(self.browser, 15)
        wait.until(EC.presence_of_all_elements_located((By.XPATH, '//div[@class = "g"]')))
    
        searchResults =[]
        a_tags =  self.browser.find_elements(By.TAG_NAME,"a")
        for tag in a_tags: 
            if re.search("google", str(tag.get_attribute("href")).lower()):
                pass
            else:
                searchResults.append(tag.get_attribute('href'))

        return list(filter(self.filter_resultLinks,searchResults))
        


    def filter_resultLinks(self,link):
        if re.match(r'^(http|https)://',str(link)):
            if(re.search("google",str(link)) or re.search("googleadservices",str(link)) or re.search("home.mobile",str(link)) or re.search("autoscout24",str(link)) or re.search("web2.cylex",str(link)) or re.search("servicesinfo",str(link)) or re.search("11880",str(link)) or re.search("facebook",str(link))):
                return
            else:
                return link
        else:
            return
    
    def save_data(self):
        all_seed_df = pd.concat([self.url_df,self.topics_df], ignore_index= True)
        print(all_seed_df)
        #all_seed_df.to_csv(os.path.join(self.package_dir,r'files\Seed.csv'), index = False)
        all_seed_df.to_feather(os.path.join(self.package_dir,r'files\Seed.feather'))

    def run(self):
        self.topics_df['URL'] = self.topics_df['Keyword'].apply(lambda row: self.get_url(row))
        self.topics_df = self.topics_df.explode('URL')
        self.save_data()
        

#Main - Init a crawler with given searchlist Searchlist.xsls. Crawls and saves all information (run).
if __name__ == "__main__":
    start = time.process_time()
    scrape = TopicScraper()
    scrape.run()
    print(time.process_time() - start)



