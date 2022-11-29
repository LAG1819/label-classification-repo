import scrapy
import pandas as pd
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
import os
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urlparse


class SeederSpider(scrapy.Spider):
    name = 'seeder'
    link_extractor = LinkExtractor()
    
    package_dir = str(os.path.dirname(__file__)).split("src")[0]
    absolute_path = os.path.join(package_dir,r'files\Seed.csv')
    seed_domain_list = pd.read_csv(absolute_path,header = 0)['URL'].tolist()
    allowed_domains = [urlparse(str(s)).netloc for s in seed_domain_list] 

    headers = {'Accept-Language': 'de;q=0.7', 
           'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

    def __init__(self, url_path,url_col, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.data = self.get_data(url_path)
        list_test = list(self.data[url_col].to_list()[0:1])
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

    def parse(self, response):
        try:
            # hxs = scrapy.Selector(response)
            # text = str(' '.join(hxs.xpath("//body//text()").extract()))
            
            reqs = requests.get(str(response.url),headers = self.headers).text
            soup = BeautifulSoup(reqs,'lxml')
            body = soup.body
            text = "|".join([str(x) for x in body.strings])
        except Exception as e:
            text = ""

        for link in self.link_extractor.extract_links(response):
            if link.url not in self.visited:
                self.visited.append(link.url)
                yield scrapy.Request(url=link.url, callback=self.parse)

        yield {
        'URL': response.url,
        'URL-TEXT':text,
        # 'SUB_URL': link.url
        }
   
def run_crawler():
    signal = True

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'FEEDS': {'files/Raw_texts.csv': {'format': 'csv'}}})
    process.crawl(SeederSpider, r'files\Seed.csv','URL')
    process.start()


    
if __name__ == '__main__':
    run_crawler()