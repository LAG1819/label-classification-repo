 #Crawls website of 'seller+city' - search for service subdirectory
    def service_search(self):
        service_links = []
        s = []

        trys = 0
        site = ''
        while site == '':
            if trys == 2:
                print("3 attempts to connect. Connection stopped")
                #print("No `Serviceleistung` was found")
                self.service_link = "Failed"
                self.subdomains = "Failed"
                self.services = "Failed"
                return
            trys += 1
            try:
                site = requests.get(self.seller_link, headers=self.headers).text
                break
            except:
                #print("Connection refused by the server.. - Wait 5 Seconds")
                time.sleep(3)
                continue


        soup = BeautifulSoup(site, 'lxml')

        # check for service hrefs on main website
        hrefs = soup.find_all("a",href = True)

        service_links = [href for href in hrefs if ((re.search("service", href['href'].lower()) or
                                                     re.search("leistung", href['href'].lower()) and not
                                                     re.search(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+', href['href'].lower())) and not
                                                     re.search("twitter", href['href'].lower()) and not
                                                     re.search("facebook", href['href'].lower())
                                                     )]

        if not service_links:
            service_links = [href for href in hrefs if (re.search("service",href.getText().lower()) or re.search("service",href["href"].lower()))]
        if not service_links:
            service_links = [href for href in hrefs if (re.search("leistung",href.getText().lower()) or re.search("leistung",href["href"].lower()))]
        if not service_links:
            service_links = [href for href in hrefs if (re.search("werkstatt",href.getText().lower()) or re.search("werkstatt",href["href"].lower()))]

        if len(service_links) > 1:
            if any("leistung" in str(link).lower() for link in service_links):
                s = [ref['href'] for ref in service_links if (re.search("leistung",ref.getText().lower())or re.search("leistung",ref["href"].lower()))]
            if (any("angebot" in str(link).lower() for link in service_links) and len(s) == 0):
                s = [ref['href'] for ref in service_links if (re.search("angebot",ref.getText().lower())or re.search("angebot",ref["href"].lower()))]
            if (any("service" in str(link).lower() for link in service_links) and len(s) == 0):
                s = [ref['href'] for ref in service_links if (re.search("service",ref.getText().lower())or re.search("service",ref["href"].lower()))]
            if not s:
                s = [ref['href'] for ref in service_links]

            s = min(s,key = len)

        elif len(service_links) == 1:
            s = service_links[0]['href']
        else:
            s = "No Service"

        try:
            self.get_subdomains(hrefs,s)
        except Exception as e:
            self.subdomains = "No Service Subdomains"


        if (re.match('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', str(s)) and s != "No Service"):
            self.service_link = str(s)
        elif (re.match('/', str(s))and s != "No Service") :
            self.service_link = str(self.seller_link) + str(s)
        elif (not re.match('/', str(s)) and s != "No Service"):
            self.service_link = str(self.seller_link) +"/" + str(s)
        elif s == "No Service":
            self.service_link = "No Service"
        else:
            self.service_link = "No Service"

    #Crawls website of 'seller+city' - crawls any navbar/menubar conatining all subdirectorys of website
    def get_subdomains(self,hrefs,s):
        domain =[href for href in hrefs if str(href['href']) == s][0]
        d_parent = domain.parent
        subdomains_found = d_parent.find_all("a", href = True)
        while len(subdomains_found) == 1:
            d_parent = d_parent.parent
            subdomains_found = d_parent.find_all("a", href = True)
        self.subdomains = ",".join([str(f) for f in subdomains_found])

    #Crawls all html text of service subdirectory  of website of 'seller+city'
    def service_retrieve(self):
        if self.service_link == "Failed":
            self.services = "Failed"
        elif self.service_link == "No Service":
            self.services = "No Service"
        else:
            reqs = requests.get(str(self.service_link), headers=self.headers).text
            soup = BeautifulSoup(reqs,'lxml')
            body = soup.body
            h = [str(x) for x in body.strings]
            h = ",".join(h)
            if ("Error" in str(h)):
                self.services = "Failed"
            else:
                self.services = h

    #Alternative approach of service_search(): Crawls website of 'seller+city' - search for service subdirectory
    def service_search_selenium(self):
        self.browser.get(self.seller_link)

        ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
        WebDriverWait(self.browser, 5,ignored_exceptions=ignored_exceptions) \
            .until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        service_links = [r for r in self.browser.find_elements_by_tag_name("a") if ((re.search("service", str(r.get_attribute("href")).lower()) or
                           re.search("leistung", str(r.get_attribute("href")).lower()) and not
                           re.search(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+', str(r.get_attribute("href")).lower())) and not
                          re.search("twitter", str(r.get_attribute("href")).lower()) and not
                          re.search("facebook", str(r.get_attribute("href")).lower()) )]
        hrefs = service_links

        if not service_links:
            service_links = [r for r in self.browser.find_elements_by_tag_name("a") if
                             (re.search("service",str(r.get_attribute("href")).lower()) or
                              re.search("service",str(r.text).lower()))]

        if not service_links:
            service_links = [r for r in self.browser.find_elements_by_tag_name("a") if
                             (re.search("leistung",str(r.get_attribute("href")).lower()) or
                              re.search("leistung",str(r.text).lower()))]

        if not service_links:
            service_links = [r for r in self.browser.find_elements_by_tag_name("a") if
                             (re.search("werkstatt",str(r.get_attribute("href")).lower()) or
                              re.search("werkstatt",str(r.text).lower()))]


        if len(service_links) > 1:
            s=[]
            if any("leistung" in str(link).lower() for link in service_links):
                s = [ref.get_attribute("href") for ref in service_links if (re.search("leistung",ref.text.lower())or re.search("leistung",ref.get_attribute("href").lower()))]

            if (any("angebot" in str(link).lower() for link in service_links) and len(s) == 0):
                s = [ref.get_attribute("href") for ref in service_links if (re.search("angebot",ref.text.lower())or re.search("angebot",ref.get_attribute("href").lower()))]

            if (any("service" in str(link).lower() for link in service_links) and len(s) == 0):
                s = [ref.get_attribute("href") for ref in service_links if (re.search("service",ref.text.lower())or re.search("service",ref.get_attribute("href").lower()))]

            if not s:
                s = [ref.get_attribute("href") for ref in service_links]

            s = min(s,key = len)

        elif len(service_links) == 1:
            s = service_links[0].get_attribute("href")
        else:
            s = "No Service"

        try:
            self.get_subdomains_selenium(hrefs,s)
        except Exception as e:
            self.subdomains = "No Service Subdomains"


        if (re.match('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', str(s)) and s != "No Service"):
            self.service_link = str(s)
        elif (re.match('/', str(s))and s != "No Service") :
            self.service_link = str(self.seller_link) + str(s)
        elif (not re.match('/', str(s)) and s != "No Service"):
            self.service_link = str(self.seller_link) +"/" + str(s)
        elif s == "No Service":
            self.service_link = "No Service"
        else:
            self.service_link = "No Service"

    #Alternative approach of get_subdomains(): Crawls website of 'seller+city' - crawls any navbar/menubar conatining all subdirectorys of website
    def get_subdomains_selenium(self,hrefs,s):
        domain =[href.get_attribute("href") for href in hrefs if str(href.get_attribute("href")) == s][0]
        d_parent = domain.find_element_by_xpath("..")
        subdomains_found = d_parent.find_elements_by_tag_name("a")
        while len(subdomains_found) == 1:
            d_parent = d_parent.find_element_by_xpath("..")
            subdomains_found = d_parent.find_elements_by_tag_name("a")
        self.subdomains = ",".join([str(f) for f in subdomains_found])

    #Alternative approach of service_retrieve(): Crawls all html text of service subdirectory  of website of 'seller+city'
    def service_retrieve_selenium(self):
        if self.service_link == "Failed":
            self.services = "Failed"
        elif self.service_link == "No Service":
            self.services = "No Service"
        else:
            self.browser.get(str(self.service_link))

            # ignored_exceptions=(NoSuchElementException,StaleElementReferenceException,)
            # WebDriverWait(self.browser, 5,ignored_exceptions=ignored_exceptions) \
            #     .until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            content = self.browser.find_element_by_tag_name("body").get_attribute("innerText")
            content = [str(x) for x in content]
            content = ",".join(content)

            if ("Error" in str(content)):
                self.services = "Failed"
            else:
                self.services = content

    #Create Dataframe of crawled Data and save as csv
    def create_dataframe_and_save(self):
        data = {'Seller': self.seller,
                'Services': self.services_all,
                'Website': self.links,
                'Website_List': self.searchLinks,
                'Service_directory':self.service_link_all,
                'Service_subdirectory':self.subdomains_all
                }

        df = pd.DataFrame(data)

        print(df.head(1))
        df.to_csv(r'../data/scraped_data.csv', index = None, header=True)

    #Create Dataframe of crawled Data
    def create_dataframe(self):
        data = {'Seller': self.seller,
                'Services': self.services_all,
                'Website': self.links,
                'Website_List': self.searchLinks,
                'Service_directory':self.service_link_all,
                'Service_subdirectory':self.subdomains_all
                }

        df = pd.DataFrame(data)
        return df