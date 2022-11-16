# import scrapy
# import pandas as pd
# from scrapy.linkextractors import LinkExtractor
# import os
# import csv


# class SeederSpider(scrapy.Spider):
#     name = 'seeder'
#     link_extractor = LinkExtractor()
#     result_links = []

#     file_name = open('Output_file.csv', 'w') 
#     fieldnames = ['URL-BASE', 'URL-FOUND'] 
#     writer = csv.DictWriter(file_name, fieldnames=fieldnames)
#     writer.writeheader()

#     def start_requests(self):
#         package_dir = str(os.path.dirname(__file__)).split("src")[0]
#         absolute_path = os.path.join(package_dir,r'files\URL_Seed.xlsx')
#         print(pd.read_excel(absolute_path,header = 0))

#         df = pd.read_excel(absolute_path,header = 0) 

#         urlList = df['URL'].to_list()
#         for url in urlList:
#              yield scrapy.Request(url = url, callback=self.parse)


#     def parse(self, response):
#         for link in self.link_extractor.extract_links(response):
#             self.result_links.append(link.url)
#             self.writer.writerow({'URL-BASE': response.url, 'URL-FOUND': link.url}) 

#         print("ALL: "+str(len(self.result_links)))# + str(self.result_links)

#     def parse(self, response):
#         hxs = scrapy.Selector(response)
#         for link in self.link_extractor.extract_links(response):
#             self.OuputSeedWriter.writerow({'URL-BASE': response.url, 'URL-FOUND': link.url}) 

#         text = str(' '.join(hxs.xpath("//body//text()").extract()))
#         # print(text)
#         # self.OuputTextWriter.writerow({'URL': response.url, 'TEXT':text})