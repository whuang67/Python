# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:47:06 2017

@author: whuang67
"""

import urllib.request as ur
from BeautifulSoup4 import *

current_repeat_count = 0
url = 'http://python-data.dr-chuck.net/known_by_Fikret.html '
repeat_count = 4
position = 3


def parse_html(url):
    html = ur.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    tags = soup('a')
    return tags

while current_repeat_count < repeat_count:
    print('Retrieving: ', url)
    tags = parse_html(url)
    for index, item in enumerate(tags):
        if index == position - 1:
            url = item.get('href', None)
            name = item.contents[0]
            break
        else:
            continue
    current_repeat_count += 1
print('Last Url: ', url)