from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
import requests
import csv
import re
import pandas as pd
with open("test.csv","a") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time","title","url"])
    #writer.writerow([[date], [newstitle], [url]])
# open the firefox and enter the html
# driver =webdriver.PhantomJS()#PhantomJS
driver =webdriver.PhantomJS(executable_path="/public/home/keyyd/phantomjs-2.1.1-linux-x86_64/bin/phantomjs")#PhantomJS
print "+++++Webpage loading!+++++"
driver.get('https://www.teslarati.com/category/news/')
print "+++++Webpage loaded!+++++"
newsTitleAll = []
newsTimeAll = []
newsURLAll = []
numberCraw = 3000
print "###Load "+str(numberCraw)+" Titles###"
for i in range(int(numberCraw/20)):
    ButtonPath = '/html/body/div[1]/div/div[3]/div[2]/div/div/div[1]/div/div[1]/div/div/div/div/a'
    driver.find_element_by_xpath(ButtonPath).click()

#import requests
#import re
for newsIndex in range(numberCraw):
    path = '/html/body/div[1]/div/div[3]/div[2]/div/div/div[1]/div/div[1]/div/div/div/div/ul/li[%d]'%(newsIndex+1)
    newstext = driver.find_element_by_xpath(path).text
    news_x = newstext.splitlines()
    newstitle = news_x[0].encode('ascii','ignore')
    newsTitleAll.append(newstitle)
    print '##title: ' + newstitle
    # MainPage: get the html of the news title and enter in it to get the date
    path = '/html/body/div[1]/div/div[3]/div[2]/div/div/div[1]/div/div[1]/div/div/div/div/ul/li[%d]'%(newsIndex+1)
    first_result = driver.find_element_by_xpath(path)
    first_link = first_result.find_element_by_tag_name('a')
    url = first_link.get_attribute("href")
    print url
    newsURLAll.append(url)
    try:       
        r=requests.get(url).text
        content =r
        res_tr = r'<time .*?>(.*?)</time>'
        m_tr =  re.findall(res_tr,content,re.S|re.M)
        # print m_tr[0]
        date = m_tr[0]
    except Exception:
        date = u'NAN'
    newsTimeAll.append(date)
    print '##date: ' + date
    # close the child page
    with open("test.csv","a") as csvfile: 
    	writer = csv.writer(csvfile)
    	# writer.writerow(["time","title","url"])
        writer.writerow([date, newstitle, url])

#newsTitleAll = [newsTitletmp.encode('ascii', 'ignore') for newsTitletmp in newsTitleAll]
#Dataframe = pd.DataFrame({'time':newsTimeAll, 'title':newsTitleAll, 'url':newsURLAll})
#Dataframe.to_csv("teslaDate_6_25.csv",index=False,sep=',')
