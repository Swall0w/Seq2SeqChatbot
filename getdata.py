import urllib.request
from bs4 import BeautifulSoup
import re
import time
import sys

def getcontents(url):
    headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64; rv:38.0) Gecko/20100101 Firefox/38.0"}
    req = urllib.request.Request(url=url,headers=headers)
    html = urllib.request.urlopen(req)
    html = html.read().decode('utf-8')
    soup = BeautifulSoup(html,'lxml')
#table = soup.findAll('span')
    return soup.find('div',id='honbun').text

def html2sentence(url):
    text = getcontents(url)
    pattern = r'[『「].+?[」』]'
    repatter = re.compile(pattern)
    matchlist = repatter.findall(text)
    sentence = []
    if matchlist:
        for match in matchlist:
#            print(match[1:-1])
            sentence.append(match[1:-1])
    return sentence

def main():
#    url= 'https://novel.syosetu.org/108541/1.html'
    original_url = 'https://novel.syosetu.org/' + str(sys.argv[1])+'/'
    index = 1
    while True:
        url= original_url + str(index) + '.html'
        sentence = html2sentence(url)
        with open('data/file'+str(index)+'.txt','w') as f:
            for text in sentence:
                f.write(text)
                f.write('\n')
        index +=1
        time.sleep(2)


if __name__ == '__main__':
    main()
