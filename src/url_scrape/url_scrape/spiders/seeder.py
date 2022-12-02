import scrapy
import pandas as pd
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
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

class SeederSpider(scrapy.Spider):
    name = 'seeder'
    link_extractor = LinkExtractor()

    headers = {'Accept-Language': 'de;q=0.7', 
           'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

    def __init__(self, url_path, name=None, **kwargs):
        super().__init__(name, **kwargs)

        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir,url_path)
        self.data = pd.read_feather(absolute_path)

        seed_list = self.data['URL'].tolist()
        #dynamically allow all domains occuring
        allowed = set() 
        for s in seed_list:
            allowed.add(urlparse(str(s)).netloc)
        self.allowed_domains = list(allowed) 

        list_test = list(seed_list[5:6])
        self.visited = list_test
        self.queue = list_test

    def get_data(self,url_path):
        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir,url_path)
        seed_df = pd.read_csv(absolute_path,header = 0) 

        return seed_df

    def start_requests(self):
        for url in self.queue:      
            yield scrapy.Request(url = url, callback=self.parse)

    def filter_link(self,link):
        pattern_list =["vimeo","apple","twitter","facebook","google","whatsapp","tiktok","pinterest", "klarna", "jobs","linkedin","xing", "mozilla","youtube", "gebrauchtwagen", "neufahrzeug"]
        flag = False
        if any(pattern in str(link).lower() for pattern in pattern_list):
            flag = True
        return flag

    def parse(self, response):
        try:
            reqs = requests.get(str(response.url),headers = self.headers).text
            soup = BeautifulSoup(reqs,'lxml')
            body = soup.body
            text = "|".join([str(x) for x in body.strings])
        except Exception as e:
            text = ""

        if text == "":
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

                # Refresh the regex cache for `allowed_domains`
                self.allowed_domains.append(urlparse(str(link.url)).netloc)

                for mw in self.crawler.engine.scraper.spidermw.middlewares:
                    if isinstance(mw, scrapy.spidermiddlewares.offsite.OffsiteMiddleware):
                        mw.spider_opened(self)
                
                if self.filter_link(link.url) == False:
                    yield scrapy.Request(url=link.url, callback=self.parse)

        yield {
        'URL': response.url,
        'URL_TEXT':text,
        }
   
def run_crawler():
    signal = True

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        # 'FEED_EXPORTERS': {
        #     'pickle': 'scrapy.exporters.PickleItemExporter'
        # },
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEEDS': {'files/raw_texts.json': {'format': 'json','encoding': 'utf8','fields': ['URL', 'URL_TEXT']}}
        })
        #{'files/raw_texts.csv': {'format': 'csv'}}
        #{'files/raw_texts.pkl': {'format': 'pickle'}}
        #{'files/raw_texts.json': {'format': 'json'}}
    process.crawl(SeederSpider, r'files\Seed.feather')
    process.start()


    
if __name__ == '__main__':
    run_crawler()
