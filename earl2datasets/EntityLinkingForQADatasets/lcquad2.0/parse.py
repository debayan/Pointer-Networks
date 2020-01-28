#!/usr/bin/python

from __future__ import print_function
import sys,json
import requests,re
from multiprocessing import Pool
import urllib.request

def hiturl(questionserial):
    question = questionserial[0]
    serial = questionserial[1]['uid']
    try:
        print(question)
        question = re.sub(r"[^a-zA-Z0-9]+", ' ', question)
        conditionsSetURL = 'https://labs.tib.eu/falcon/falcon2/api?mode=short'
        newConditions = {'text': question}
        params = json.dumps(newConditions).encode('utf8')
        req = urllib.request.Request(conditionsSetURL, data=params, headers={'content-type': 'application/json'})
        response = urllib.request.urlopen(req)
        response = response.read().decode('utf8')
        print(response)
        return (serial,response,questionserial[1])
    except Exception as e:
        return(serial,'[]',questionserial[1])

f = open('LC-QuAD2.0/dataset/test.json')
s = f.read()
d = json.loads(s)
f.close()
questions = []

for item in d:
    questions.append((item['question'],item))

pool = Pool(3)
responses = pool.imap(hiturl,questions)

_results = []

count = 0
totalentchunks = 0
tpentity = 0
fpentity = 0
fnentity = 0
for response in responses:
    count += 1
    print(count)
    _results.append((response[0],json.loads(response[1])))

#_results = sorted(_results, key=lambda tup: tup[0])

results = []
for result in _results:
    results.append(result)

 
f1 = open('falcon2lcqtest.json','w')
print(json.dumps(results),file=f1)
f1.close()
