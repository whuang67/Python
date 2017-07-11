# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:51:28 2017

@author: whuang67
"""

import urllib
import xml.etree.ElementTree as ET

URL = 'http://python-data.dr-chuck.net/comments_329332.xml'

data = urllib.request.urlopen(URL).read()
tree = ET.fromstring(data)
counts = tree.findall('comments/comment/count')

total = 0
for count in counts:
    total = total+int(count.text)
    
print(total)
