#amazon scraper nasdaq
import requests
from bs4 import BeautifulSoup
URL = "https://www.nasdaq.com/symbol/amzn/real-time"
#URL = "https://www.nasdaq.com/symbol/googl/real-time"
#URL = "https://www.nasdaq.com/symbol/ta/real-time"
amazon = requests.get(URL)
soup = BeautifulSoup(amazon.content,'html.parser')
netchange = soup.find("div",id = 'qwidget_netchange')
#netchange = soup.find('div',class_ = 'qwidget-cents qwidget-Green')
#print(type(netchange))
lastsale = soup.find('div',id='qwidget_lastsale')
print("Last sale: " + lastsale.get_text())
print("Net Change is : " + netchange.get_text())
if soup.find('div',class_="marginLR10px arrow-green"): 
    print("1")
else:
    print("0")