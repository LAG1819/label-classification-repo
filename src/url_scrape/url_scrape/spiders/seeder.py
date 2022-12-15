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

    def __init__(self, lang:str, name=None, **kwargs):
        """Initalisation of WebCrawler. 

        Args:
            lang (str): unicode to select texts in that language 
            name (_type_, optional): Name of the scrapy crawler. Defaults to None.
        """
        super().__init__(name, **kwargs)
        self.lang = lang

        self.headers = {'Accept-Language': 'de;q=0.7', 
            'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

        self.queue = self.get_data()[['CLASS','URL']].values.tolist()
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
            yield scrapy.Request(url = url[1], meta={'topic': url[0]},callback=self.parse)

    def get_data(self):
        """Get the input data with the url links to be retrieved.

        Returns:
            DataFrame: Pandas DataFrame containing the columns CLASS(class labels), URL(url links for retrival), LANG(language), 
        """
        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir, r'files\TOPIC_Classes.xlsx')
        seed_df = pd.read_excel(absolute_path,header = 0)
        seed_df =  seed_df[seed_df['LANG']==self.lang] 
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
        'CLASS':response.meta['topic'],
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

        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir,url_path)
        self.data = pd.read_feather(absolute_path)

        self.parse_style = parse

        # add all domains of input urls to list of allowed domains to crawl
        seed_list = self.data['URL'].tolist()
        allowed = set() 
        for s in seed_list:
            allowed.add(urlparse(str(s)).netloc)
        self.allowed_domains = list(allowed) 

        list_test = list(seed_list[0:3])
        self.visited = list_test
        self.queue = list_test

    def get_data(self,url_path):
        """Get the input data with the url links to be retrieved.

        Returns:
            DataFrame: Pandas DataFrame containing the columns CLASS(class labels), URL(url links for retrival), LANG(language), 
        """
        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir,url_path)
        seed_df = pd.read_csv(absolute_path,header = 0) 
        return seed_df

    def start_requests(self):
        """Start of Retrieval of the individual input url links. Scrapy own mandatory function.

        Yields:
            request : calling the scrapy own request query 
        """
        for url in self.queue:      
            yield scrapy.Request(url = url, callback=self.parse)
        
        
    def filter_link(self,link):
        """

        Args:
            link (str): Scraped link to be checked with given blacklist.

        Returns:
            boolean: Returns true if domain of given link is in blacklist.
        """
        black_list =["foerderland","umweltbundesamt","ihk","capital","marketing","billiger","instagram","spotify","deezer","shop","configure","github","vimeo","apple","twitter","facebook","google","whatsapp","tiktok","pinterest", "klarna", "jobs","linkedin","xing", "mozilla","youtube", "gebrauchtwagen", "neufahrzeug"]
        flag = False
        if any(pattern in str(link).lower() for pattern in black_list):
            flag = True
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
        try:
            reqs = requests.get(str(response.url),headers = self.headers).text
            soup = BeautifulSoup(reqs,'lxml')
            body = soup.body
            text = "|".join([str(x) for x in body.strings])
        except Exception as e:
            text = "javascript"

        if ("javascript" in text) or ("java script" in text) or ('Access denied' in text):
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

        for link in self.link_extractor.extract_links(response):
            if link.url not in self.visited:
                self.visited.append(link.url)                
                #Filter found links on website based on predfined blacklist
                if self.filter_link(link.url) == False:
                    #dynamically allow all domains occuring
                    if self.parse_style == "internal":
                        yield scrapy.Request(url=link.url, callback=self.parse)
                    else:
                        # Refresh the regex cache for `allowed_domains`
                        self.allowed_domains.append(urlparse(str(link.url)).netloc)

                        for mw in self.crawler.engine.scraper.spidermw.middlewares:
                            if isinstance(mw, scrapy.spidermiddlewares.offsite.OffsiteMiddleware):
                                mw.spider_opened(self)

                        yield scrapy.Request(url=link.url, callback=self.parse)

        yield {
        'DOMAIN':str(urlparse(str(response.url)).netloc),
        'URL': response.url,
        'URL_TEXT':text,
        }

def union_data():
    """Combines all external and internal crawled websites (based on the specified domains). Duplicates are removed. Saves all crawled websites under raw_texts.json.
    """
    df_path_i = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts_internal.json"
    df_path_e = str(os.path.dirname(__file__)).split("src")[0] + r"files\raw_texts_external.json"
    internal_data = pd.read_json(df_path_i)
   
    try:
        external_data = pd.read_json(df_path_e)
        outer = external_data.merge(internal_data, how='outer', indicator=True)
        merged_df = pd.concat([internal_data,external_data]).drop_duplicates(subset = 'URL', keep = 'first').reset_index(drop=True)
        #external_data_anti_joined = outer[(outer._merge=='left_only')].drop('_merge', axis=1)
    except Exception as e:
        merged_df = internal_data

    merged_df.to_json(r'files\raw_texts.json', orient = 'records')

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
        'FEEDS': {'files/raw_texts_internal.json': {'format': 'json','encoding': 'utf8','fields': ["DOMAIN",'URL', 'URL_TEXT']}}
        })
        #{'files/raw_texts.csv': {'format': 'csv'}}
        #{'files/raw_texts.pkl': {'format': 'pickle'}}
        #{'files/raw_texts.json': {'format': 'json'}}

    process2 =  CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEEDS': {'files/raw_texts_external.json': {'format': 'json','encoding': 'utf8','fields': ["DOMAIN",'URL', 'URL_TEXT']}}
        })

    process3 =  CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEEDS': {'files/raw_classes.json': {'format': 'json','encoding': 'utf8','fields': ['CLASS','DOMAIN','URL', 'URL_TEXT']}}
        })

    process.crawl(SeederSpider, r'files\Seed.feather', 'internal')
    process2.crawl(SeederSpider, r'files\Seed.feather', 'external')
    # process3.crawl(TopicSpider,'DE')
    process.start()
    process2.start()
    # process3.start()
    
if __name__ == '__main__':
    """Main function. Calls run method "run_crawler" and union method "union_data"
    """
    run_crawler()
    union_data()
