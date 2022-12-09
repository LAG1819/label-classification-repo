import scrapy
import pandas as pd
import os
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

class TopicSpider(scrapy.Spider):
    name = 'topic'
    link_extractor = LinkExtractor()

    def __init__(self, lang, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.lang = lang

        self.headers = {'Accept-Language': 'de;q=0.7', 
            'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

        self.queue = self.get_data()[['TOPIC','URL']].values.tolist()
        print(self.queue)
        #dynamically allow all domains occuring
        allowed = set() 
        for s in self.queue:
            allowed.add(urlparse(str(s[1])).netloc)
        self.allowed_domains = list(allowed) 

    def start_requests(self):
        for url in self.queue:
            yield scrapy.Request(url = url[1], meta={'topic': url[0]},callback=self.parse)

    def get_data(self):
        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir, r'files\TOPIC_text.xlsx')
        seed_df = pd.read_excel(absolute_path,header = 0)
        seed_df =  seed_df[seed_df['LANG']==self.lang] 
        return seed_df

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
                
        yield {
        'TOPIC':response.meta['topic'],
        'URL': response.url,
        'URL_TEXT':text,
        }
   
def run_crawler():
    signal = True

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'FEED_EXPORT_ENCODING': 'utf-8',
        'FEEDS': {'files/topic_texts.json': {'format': 'json','encoding': 'utf8','fields': ['TOPIC','URL', 'URL_TEXT']}}
        })

    process.crawl(TopicSpider,'DE')
    process.start()

if __name__ == '__main__':
    run_crawler()
