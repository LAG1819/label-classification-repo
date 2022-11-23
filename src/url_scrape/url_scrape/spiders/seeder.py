import scrapy
import pandas as pd
from scrapy.linkextractors import LinkExtractor
import os
from bs4 import BeautifulSoup
import requests


class SeederSpider(scrapy.Spider):
    name = 'seeder'
    link_extractor = LinkExtractor()
    package_dir = str(os.path.dirname(__file__)).split("src")[0]
    headers = {'Accept-Language': 'de;q=0.7', 
           'User-agent':"Mozilla/101.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0. 5005.78 Safari/537.36 Edge/100.0.1185.39"}

    base_links = []
    result_links = []
    result_text =[]
    url_text = []
    result_text_lang =[]

    def start_requests(self):
        absolute_path = os.path.join(self.package_dir,r'files\URL_Seed.xlsx')
        print(pd.read_excel(absolute_path,header = 0))

        df = pd.read_excel(absolute_path,header = 0) 

        urlList = df['URL'].to_list()#[1:5]
        for url in urlList:
             yield scrapy.Request(url = url, callback=self.parse)


    def parse(self, response):
        for link in self.link_extractor.extract_links(response):
            self.result_links.append(link.url)
            self.base_links.append(response.url)

        seeds_df = pd.DataFrame(data = {'URL': self.base_links, 'URL-FOUND': self.result_links})
        seeds_df.to_csv(os.path.join(self.package_dir,r'files\URL_Subseed.csv'), index = False)
        print("ALL: "+str(len(self.result_links)))# + str(self.result_links)

        # hxs = scrapy.Selector(response)
        # text = str(' '.join(hxs.xpath("//body//text()").extract()))
        
        try:
            reqs = requests.get(str(response.url),headers = self.headers).text
            soup = BeautifulSoup(reqs,'lxml')
            body = soup.body
            text = "|".join([str(x) for x in body.strings])
        except Exception as e:
            text = ""

        #print(text)

        self.result_text.append(text)
        self.url_text.append(response.url)

        text_df = pd.DataFrame(data = {'URL': self.url_text, 'TEXT': self.result_text})
        text_df.to_csv(os.path.join(self.package_dir,r'files\Output_texts.csv'), index = False)