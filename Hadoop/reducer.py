# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 08:17:55 2017

@author: whuang67
"""

import sys
def reducer():
    salesTotal= 0
    oldKey = None
    
    for line in sys.stdin:
        data = line.strip().split("\t")
        
        if len(data) != 2:
            continue
        
        thisKey, thisSale = data
        
        if oldKey and oldKey != thisKey:
            print("{}\t{}".format(oldKey, salesTotal))
            salesTotal = 0
        
        salesTotal += float(thisSale)
        oldKey = thisKey
    
    if oldKey:
        print("{}\t{}".format(oldKey, salesTotal))


test_text = """Miami\t12.34
Miami\t99.07
Miami\t3.14
NYC\t99.77
NYC\t88.99
test\t1.22
test\t100.2
"""

def main():
	from io import StringIO
	sys.stdin = StringIO(test_text)
	reducer()
	sys.stdin = sys.__stdin__

if __name__ == '__main__':
    main()