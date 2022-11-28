import scrapy
import pandas as pd
from scrapy.linkextractors import LinkExtractor
from scrapy.crawler import CrawlerProcess
import os
from bs4 import BeautifulSoup
import requests


class SeederSpider(scrapy.Spider):
    name = 'seeder'
    link_extractor = LinkExtractor()
    headers = {'Accept-Language': 'de;q=0.7', 
           'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

    input_links = []
    found_links = []
    found_links_list =[]
    result_text =[]
    url_text = []
    result_text_lang =[]

    def __init__(self, url_path,url_col, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.data, self.text_data = self.get_data(url_path)
        self.url_list = self.data[url_col].to_list()[0:1]

    def get_data(self,url_path):
        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        absolute_path = os.path.join(package_dir,url_path)
        seed_df = pd.read_csv(absolute_path,header = 0) 

        try:
            path = os.path.join(package_dir,r'files\Output_texts.csv')
            text_df = pd.read_csv(path,header = 0) 
        except:
            text_df = pd.DataFrame()
        return seed_df,text_df

    def start_requests(self):
        for url in self.url_list:
             yield scrapy.Request(url = url, callback=self.parse)

    def parse(self, response):
        for link in self.link_extractor.extract_links(response):
            yield {
            'URL': response.url,
            'SUB_URL': link.url
            }
        
        try:
            # hxs = scrapy.Selector(response)
            # text = str(' '.join(hxs.xpath("//body//text()").extract()))
            
            reqs = requests.get(str(response.url),headers = self.headers).text
            soup = BeautifulSoup(reqs,'lxml')
            body = soup.body
            text = "|".join([str(x) for x in body.strings])
        except Exception as e:
            text = ""

        self.result_text.append(text)
        self.url_text.append(response.url)

        package_dir = str(os.path.dirname(__file__)).split("src")[0]
        new_text_df = pd.DataFrame(data = {'URL': self.url_text, 'TEXT': self.result_text})
        all_text_df = pd.concat([self.text_data,new_text_df]).drop_duplicates(subset=['TEXT'])
        all_text_df.to_csv(os.path.join(package_dir,r'files\Output_texts.csv'), index = False)  
 
def check_if_urls():
    package_dir = str(os.path.dirname(__file__)).split("src")[0]
    path = os.path.join(package_dir,r'Input_url.csv')
    df = pd.read_csv(path, header=0)
    if df.empty:
        return False
    else:
        return True


def run_crawler():
    signal = True

    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'FEEDS': {'Input_url.csv': {'format': 'csv'}}})
    process.crawl(SeederSpider, r'files\Seed.csv','URL')
    process.start(stop_after_crawl=False)

    
    while signal:
        process_sub = CrawlerProcess({
            'USER_AGENT': 'Mozilla/5.0',
            'FEEDS': {'Input_url.csv': {'format': 'csv'}}})
        process_sub.crawl(SeederSpider, r'Input_url.csv', 'SUB_URL')
        process_sub.start(stop_after_crawl=False)

        signal = check_if_urls()
        
    
if __name__ == '__main__':
    run_crawler()