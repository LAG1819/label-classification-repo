# <one line to give the program's name and a brief idea of what it does.>
# Copyright (C) 2023  Luisa-Sophie Gloger

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from bs4 import BeautifulSoup
from urllib.parse import urlencode, urlunparse
from urllib.request import urlopen, Request
import urllib3
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
import certifi

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['GH_TOKEN'] = s.github_token
os.environ['WDM_LOG'] = str(logging.NOTSET)
os.environ['WDM_LOCAL'] = '1'
os.environ['WDM_SSL_VERIFY'] = '0'

# Webcrawler - crawls for website and service information
# Saves information in a csv. No specific Return Value.
# :param searchlist: a list of strings containing seller+city per item
# :type searchlist: list
class TopicScraper:
    """This Crawler takes the file TOPIC_Seed.xlsx and crawls all keywords given in that file. 
        For each keyword the top 10 result urls of a google search with the dedicated keywords is saved. 
        The result urls are saved together with all found urls in URL_Seed.xlsx as Seed.feather.
        Seed.feather contains the initial set of url links that form the basis (seed) for the overall dataset.
    """
  
    def __init__(self, lang:str, s_path:str):
        """Initialisation of Topic Crawler. Sets Selenium Browser and calls load_data. 

        Args:
            lang (str): unicode of language to select column with keywords only in that language.
            s_path (str): Source Path of file to be loaded as seed file. 
        """
        self.headers = {'Accept-Langugage':'de;q=0.7',
                   'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Edge/100.01185.39"}
        self.http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            ca_certs=certifi.where()
            )

        self.lang = lang   
        self.package_dir = str(os.path.dirname(__file__)).split("src")[0]
        self.topics_df, self.url_df = self.load_data(s_path)

        self.options = webdriver.FirefoxOptions()
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--headless')
        self.options.add_argument(f'user-agent={self.headers}')
        self.browser = webdriver.Firefox(service=Service(GeckoDriverManager().install()),options=self.options)
        self.wait = WebDriverWait(self.browser, 20)     

    def load_data(self, s_path:str):
        """Call of both Seed files. TOPIC_Seed.xlsx contains keywords to crawl. URL_Seed.xlsx contains links to crawl.

        Args:
            s_path (str): Source Path of file to be loaded as seed file. 

        Returns:
            DataFrame: Return loaded predefined topic(keywords) and url links each as pandas DataFrame.  
        """
        absolute_path = os.path.join(self.package_dir,s_path)
        seed_data = pd.read_excel(absolute_path,header = 0) 
        
        url_col = 'URL_'+self.lang.upper()
        url_df = seed_data[['CLASS_K','KEYWORD', url_col]]
        url_df = url_df.rename(columns={url_col:'URL', 'CLASS_K':'CLASS'})
        url_df = url_df[['CLASS','KEYWORD', 'URL']].dropna()

        keyword_col = "KEYWORD_"+ self.lang.upper()
        topic_df = seed_data[['CLASS', keyword_col]]
        topic_df = topic_df.rename(columns={keyword_col:'KEYWORD'})
        topic_df = topic_df[['CLASS', 'KEYWORD']].dropna()
        
        return topic_df,url_df 
        
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
        searchRefs =[]
        for sr in searchResults:
            a_elements = sr.find_all("a")
            for a in a_elements:
                searchRefs.append(a.get('href')) 
        #searchRefs = [sr.find("a")["href"] for sr in searchResults] Not working bc of calling all a is needed before getting href is possible
        filteredsearchRefs= list(filter(self.filter_resultLinks,searchRefs))
        #print(filteredsearchRefs)

        return filteredsearchRefs

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
            pattern_list = ["google","googleadservices","home.mobile","autoscout24","web2.cylex","servicesinfo","11880",\
                "wikipedia","accessor","hotel","musical","boutique","bafa","media","photo","foto","file","europa.eu","order","gewinnspiel",\
                    "conditions","terms","legal","subscription","abonn","cooky","cookie","policy","rechtlich","privacy","datenschutz","suche",\
                        "formular", "pdf","foerderland","umweltbundesamt","ihk","capital","marketing","billiger","instagram","spotify","deezer","shop","github",\
                            "vimeo","apple","twitter","facebook","google","whatsapp","tiktok","pinterest", "klarna", "jobs","linkedin","xing", "mozilla","youtube",\
                                "store", "update"]
            if not any(re.search(pattern,str(link)) for pattern in pattern_list):
                return link
        else:
            return
    
    def save_data(self):
        """Concatenates dataframe containing all google search result website links based on keywords withdataframe containing dataframe containing website links.
        """
        all_seed_df = pd.concat([self.url_df,self.topics_df], ignore_index= True)
        all_seed_df = all_seed_df[all_seed_df[["CLASS","KEYWORD","URL"]].notnull()].reset_index(drop=True)

        print(all_seed_df.columns)
        print(all_seed_df.shape)
        # print(all_seed_df)
        
        if self.lang == 'de':
            if os.path.exists(os.path.join(self.package_dir,r'files\Seed.feather')):
                os.remove(os.path.join(self.package_dir,r'files\Seed.feather'))
            all_seed_df.to_feather(os.path.join(self.package_dir,r'files\Seed.feather'))
        else:
            if os.path.exists(os.path.join(self.package_dir,r'files\Seed_en.feather')):
                os.remove(os.path.join(self.package_dir,r'files\Seed_en.feather'))
            all_seed_df.to_feather(os.path.join(self.package_dir,r'files\Seed_en.feather'))

    def run(self):
        """Run method of class. Applies rowwise google search on given keywords and saves top 10 result website links in new column "URL". 
        Saves dataframe containing all google search result website links based on keywords with other dataframe containing website links.
        """
        self.topics_df['URL'] = self.topics_df['KEYWORD'].apply(lambda row: self.get_url(row))
        self.topics_df = self.topics_df.explode('URL')
        self.save_data()
        

#Main - Init a crawler with given searchlist Searchlist.xsls. Crawls and saves all information (run).
# if __name__ == "__main__":
#     start = time.process_time()
#     scrape = TopicScraper("en",r'files\Seed.xlsx')
#     scrape.run()
#     print(time.process_time() - start)



