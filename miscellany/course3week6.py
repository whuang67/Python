# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:32:59 2017

@author: whuang67
"""

import json
import urllib

#URL = 'http://python-data.dr-chuck.net/comments_329336.json'
#connection = urllib.request.urlopen(URL)
#input = connection.read().decode('utf-8')

#info = json.loads(input)
#print('User count:', len(info))


#count = []
#for item in info['comments']:
#    count.append(int(item['count']))
#print(sum(count))



urlservice = 'http://python-data.dr-chuck.net/geojson?'

address = 'NYU'
URL1 = urlservice + urllib.parse.urlencode({'sensor': 'false', 'address':address})
data = urllib.request.urlopen(URL1).read().decode('utf-8')
info1 = json.loads(data)
print(info1['results'][0]['place_id'])
