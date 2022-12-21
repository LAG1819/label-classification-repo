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

import scrapy
import pandas as pd
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
import os
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

class TopicSpider(scrapy.Spider):
    """WebCrawler that retrieves the URLs used as future topics from the manually specified classes in TOPIC_Classes.xlsx file,
    retrieves the website text of these and saves them in a new raw_classes.json file.

    Args:
        scrapy (Spider): Scrapy Spider Object

    Returns:
        None : None

    Yields:
        json : raw_classes.json  
    """
    name = 'topic'
    link_extractor = LinkExtractor()

    def __init__(self, lang:str,t_path:str, name=None, **kwargs):
        """Initalisation of WebCrawler. 

        Args:
            lang (str): unicode to select texts in that language 
            t_path (str): Target path were crawled data will be stored at.
            name (_type_, optional): Name of the scrapy crawler. Defaults to None.
        """
        super().__init__(name, **kwargs)
        self.lang = lang
        self.target_path = t_path

        self.headers = {'Accept-Language': 'de;q=0.7', 
            'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

        self.queue = self.get_data().values.tolist()
        print(self.queue)
        #dynamically allow all domains occuring
        allowed = set() 
        for s in self.queue:
            allowed.add(urlparse(str(s[1])).netloc)
        self.allowed_domains = list(allowed) 

    def start_requests(self):
        """Start of Retrieval of the individual input url links. Scrapy own mandatory function.

        Yields:
            request : calling the scrapy own request query 
        """
        for url in self.queue:
            yield scrapy.Request(url = url[1], meta={'CLASS': url[0]},callback=self.parse)

    def get_data(self):
        """Get the input data with the url links to be retrieved. Deletes existing ouput file where crawled data will be saved to.

        Returns:
            DataFrame: Pandas DataFrame containing the columns CLASS(class labels), URL(url links for retrival), LANG(language), 
        """
        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir, r'files\Seed.xlsx')
        
        if os.path.exists(package_dir+self.target_path):
            os.remove(package_dir+self.target_path)

        seed_df = pd.read_excel(absolute_path,header = 0)[['KMEANS_CLASS', 'KMEANS_URL', 'KMEANS_LANG']]
        seed_df = seed_df.rename(columns={"KMEANS_CLASS": "CLASS", 'KMEANS_URL':'URL','KMEANS_LANG':'LANG'})
        seed_df = seed_df[['CLASS','URL', 'LANG']].dropna()
        seed_df = seed_df[seed_df['LANG']==self.lang]

        return seed_df

    def parse(self, response):
        """Scrapy mandatory function. Actual parsing of the retrieved website. Parses text content of requested url with help of BeautifoulSoup and Selenium. 

        Args:
            response (dictionary): Response of the requested query.

        Yields:
            json: raw_classes.json
        """
        try:
            reqs = requests.get(str(response.url),headers = self.headers).text
            soup = BeautifulSoup(reqs,'lxml')
            body = soup.body
            text = "|".join([str(x) for x in body.strings])
        except Exception as e:
            text = "javascript"

        if ("javascript" in text) or ("java script" in text)or ('Access denied' in text):
            try:
                browser = webdriver.Firefox(service=Service(GeckoDriverManager().install()),options=self.options)
                browser.get(response.url)

                ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
                WebDriverWait(self.browser, 3,ignored_exceptions=ignored_exceptions) \
                    .until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                content = self.browser.find_element_by_tag_name("body").get_attribute("innerText")
                text = "|".join([str(x) for x in content])
            except:
                text =""
                
        yield {
        'CLASS':response.meta['CLASS'],
        'DOMAIN':str(urlparse(str(response.url)).netloc),
        'URL': response.url,
        'URL_TEXT':text,
        }

class SeederSpider(CrawlSpider):
    """WebCrawler that retrieves all seed urls (Seed.feather), fetches the website text of these and saves it in a new raw_texts_internal.json/ raw_texts_external.json file.

    Args:
        scrapy (Spider): Scrapy Spider Object

    Returns:
        None : None

    Yields:
        json : raw_classes.json 
    """
    name = 'seeder'
    link_extractor = LinkExtractor()
    # custom_settings = {
    #     'DEPTH_LIMIT': 2,
    # }

    headers = {'Accept-Language': 'de;q=0.7', 
           'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

    def __init__(self, url_path:str,parse:str, name=None, **kwargs):
        """Initalisation of WebCrawler.

        Args:
            url_path (str): path to file containing url links to crawl
            parse (str): type of crawling
            name (_type_, optional): Name of the scrapy crawler. Defaults to None.
        """
        super().__init__(name, **kwargs)

        self.data = self.get_data(url_path)

        self.parse_style = parse

        # add all domains of input urls to list of allowed domains to crawl
        seed_list = self.data['URL'].tolist()
        allowed = set() 
        for s in seed_list:
            allowed.add(urlparse(str(s)).netloc)
        self.allowed_domains = list(allowed) 

        self.visited = []
        self.get_already_visited()
        
        self.queue = self.data[['CLASS','KEYWORD','URL']].values.tolist()
        # print(self.queue)
        # print(self.visited)

    def get_data(self,url_path):
        """Get the input data with the url links to be retrieved.

        Returns:
            DataFrame: Pandas DataFrame containing the columns CLASS(class labels), URL(url links for retrival), LANG(language), 
        """
        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir,url_path)
        seed_df = pd.read_feather(absolute_path).dropna()
        return seed_df

    def get_already_visited(self):
        """Because of possible long crawling times depending on the size of the seed, crawling can be interrupted because of lost internet connection etc.. 
        Because of that this helper function is created to link to already crawled content and not to crawl again the already crawled pages.
        """
        df_path_i = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts_internal.json"
        df_path_e = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts_external.json"
        
        if self.parse_style == 'internal':
            if os.path.exists(df_path_i):
                visited = pd.read_json(df_path_i,orient = 'records')['URL'].tolist()
                self.visited += visited
        elif self.parse_style == 'external':
            if os.path.exists(df_path_e):
                visited = pd.read_json(df_path_e,orient = 'records')['URL'].tolist()
                self.visited += visited

        self.visited =list(set(self.visited))

    def start_requests(self):
        """Start of Retrieval of the individual input url links. Scrapy own mandatory function.

        Yields:
            request : calling the scrapy own request query 
        """
        for url in self.queue:      
            yield scrapy.Request(url = url[2], meta={'CLASS': url[0], 'KEYWORD':url[1]},callback=self.parse)
        
        
    def filter_link(self,link:str, link_text:str) -> bool:
        """This helper functions filters an input link name and its text description. It checks if the link contains any word of a defined blacklist and if it is not
        a link with ending .de, .com or .en. If one of the criteria is met, the link is marked (flag = True) and thus placed on the list of pages not to be crawled.
        Otherwise flag = False and the website is free to be crawled.

        Args:
            link (str): A url of form "https:\\www...". 
            link_text (str): A given description of related link.

        Returns:
            bool: Return a boolean "flag". If True is returned the given link will not be crawled, if False is returned the given link will be crawled. 
        """
        black_list =["wikipedia","bundesregierung","accessor","hotel","musical","boutique","bafa","media","photo","foto","file","europa.eu","order","gewinnspiel","conditions","terms","legal","subscription","abonn","cooky","cookie","policy","rechtlich","privacy","datenschutz","suche",\
            "formular", "pdf","foerderland","umweltbundesamt","ihk","capital","marketing","billiger","instagram","spotify","deezer","shop","github",\
                "vimeo","apple","twitter","facebook","twitch","google","whatsapp","tiktok","pinterest", "klarna", "jobs","linkedin","xing", "mozilla","youtube",\
                    "gebrauchtwagen", "neufahrzeug","garage","rent", "impressum", "imprint", "masthead", "newsletter", "kontakt", "contact", "karriere", "career", "login",\
                        "termin", "store", "update", "accessor"]
        journals = ["spiegel", "sueddeutsche", "handelsblatt", "faz"]
        selected_countryURL = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.(de|com|en)\\b(?:[-a-zäöüßA-Z0-9()@:%_\\+.~#?&\\/=]*)$"

        flag = False
        #check blacklist in link or link description
        if any(pattern in link.lower() for pattern in black_list) or any(pattern in link_text.lower() for pattern in black_list):
            flag = True

        #check if journal websites like spiegel.de are in category automotive
        if any(j in link.lower() for j in journals):
            branche = ['auto', "car", "mobilitaet", "mobilität", "mobility"]
            if not any(b in link.lower() for b in branche):
                flag = True

        # if "wikipedia" in link.lower():
        #     if not "https://de.wikipedia.org/wiki/Portal:Auto_und_Motorrad"

        # if not re.search(selected_countryURL,link.lower()):
        #     flag = True
        # if not re.search(selected_countryURL,link_text.lower()):
        #     flag = True
        return flag

    def parse(self, response):
        """Scrapy mandatory function. Actual parsing of the retrieved website. Parses text content of requested url with help of BeautifoulSoup and Selenium. 
        Dynamically adds all urls found on the website to the list of urls to be crawled (Breadth-first-search). A distinction is made between internal and external crawling. 
        In ase of external crawling, newly identified domains are added to the list of allowed domains to be crawled.

        Args:
            response (dictionary): Response of the requested query.

        Yields:
            json: raw_texts_internal.json / raw_texts_external.json
        """
        if str(response.url) not in self.visited:
            try:
                reqs = requests.get(str(response.url),headers = self.headers).text
                soup = BeautifulSoup(reqs,'lxml')
                body = soup.body
                text = "|".join([str(x) for x in body.strings])
            except Exception as e:
                text = "javascript"

            if ("javascript" in text.lower()) or ("java script" in text.lower()) or ('denied' in text.lower()):
                try:
                    browser = webdriver.Firefox(service=Service(GeckoDriverManager().install()),options=self.options)
                    browser.get(response.url)

                    ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
                    WebDriverWait(self.browser, 3,ignored_exceptions=ignored_exceptions) \
                        .until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                    content = self.browser.find_element_by_tag_name("body").get_attribute("innerText")
                    text = " ".join([str(x) for x in content])
                except:
                    text =""
        
            yield {
            'CLASS':response.meta['CLASS'],
            'KEYWORD':response.meta['KEYWORD'],
            'DOMAIN':str(urlparse(str(response.url)).netloc),
            'URL': response.url,
            'URL_TEXT':text,
            }   

            self.visited.append(str(response.url))             
        #print(self.link_extractor.extract_links(response))
        for link in self.link_extractor.extract_links(response):
            if str(link.url) in self.visited:
                continue
            else:             
                #Filter found links on website based on predfined blacklist
                if self.filter_link(str(link.url), str(link.text)) == False:
                    if self.parse_style == "internal":
                        yield scrapy.Request(url=link.url, meta={'CLASS': response.meta['CLASS'], 'KEYWORD':response.meta['KEYWORD']},callback=self.parse)
                    #dynamically allow all domains occuring if an external search is triggered
                    else:
                        # Refresh the regex cache for `allowed_domains`
                        self.allowed_domains.append(urlparse(str(link.url)).netloc)

                        for mw in self.crawler.engine.scraper.spidermw.middlewares:
                            if isinstance(mw, scrapy.spidermiddlewares.offsite.OffsiteMiddleware):
                                mw.spider_opened(self)

                        yield scrapy.Request(url=link.url, meta={'CLASS': response.meta['CLASS'], 'KEYWORD':response.meta['KEYWORD']}, callback=self.parse)

def run_crawler():
    """
    Run method that generates 3 processes with one Scrapy Crawler each. Each process defines the ouput file to be saved.
    Process 1 generates a crawler that retrieves all domain-specific urls based on the input(seed)-links/domains.
    Process 2 generates a crawler that retrieves all urls based on the input url-links/domains, as well as domain-specific and newly found urls with other domains.
    Process 3 generates a crawler that, based on the input url-links/domains, retrieves all domain-specific urls that contain purely information on predefined classes. 
    The 3 crawlers run parallel to each other. 
    """
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        # 'FEED_EXPORTERS': {
        #     'pickle': 'scrapy.exporters.PickleItemExporter'
        # },
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEEDS': {'files/raw_texts_internal.json': {'format': 'json','encoding': 'utf8','fields': ["CLASS", "KEYWORD","DOMAIN",'URL', 'URL_TEXT']}}
        })
        #{'files/raw_texts.csv': {'format': 'csv'}}
        #{'files/raw_texts.pkl': {'format': 'pickle'}}
        #{'files/raw_texts.json': {'format': 'json'}}

    process2 =  CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEEDS': {'files/raw_classes.json': {'format': 'json','encoding': 'utf8','fields': ["CLASS", "KEYWORD",'DOMAIN','URL', 'URL_TEXT']}}
        })

    process3 = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        # 'FEED_EXPORTERS': {
        #     'pickle': 'scrapy.exporters.PickleItemExporter'
        # },
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEEDS': {'files/raw_texts_external.json': {'format': 'json','encoding': 'utf8','fields': ["CLASS", "KEYWORD","DOMAIN",'URL', 'URL_TEXT']}}
        })

    # process2.crawl(TopicSpider,'DE', r'files/raw_classes.json')   
    # process2.start()
    process.crawl(SeederSpider, r'files\Seed.feather', 'internal')
    process3.crawl(SeederSpider, r'files\Seed.feather', 'external')
    process.start()   
    process3.start()   
    
    
if __name__ == '__main__':
    """Main function. Calls run method "run_crawler" and union method "union_data"
    """
    run_crawler()

    # df_path_i = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts_internal.json"
    # df_path_e = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts_external.json"
    # dfi = pd.read_json(df_path_i,orient = 'records')
    # dfi = dfi.drop(dfi[~dfi['URL'].str.contains("boutique")].index)
    # os.remove(df_path_i)
    # dfi.to_json(df_path_i, orient = 'records')
    
    # dfe = pd.read_json(df_path_e,orient = 'records')
    # dfe = dfe.drop(dfe[~dfe['URL'].str.contains("boutique")].index)
    # os.remove(df_path_e)
    # dfe.to_json(df_path_e, orient = 'records')
    # print("Saved data")