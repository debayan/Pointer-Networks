#!/usr/bin/python

from __future__ import print_function
import sys,json
import requests,re
from multiprocessing import Pool
import urllib.request

def hiturl(questionserial):
    question = questionserial[0]
    serial = questionserial[1]['question_id']
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

f = open('input/webqsp.test.entities.with_classes.json')
s = f.read()
d = json.loads(s)
f.close()
questions = []

for item in d:
    questions.append((item['utterance'],item))

pool = Pool(5)
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
#    item = response[2]
#    goldentities = re.findall( r'wd:(.*?) ', item['sparql_wikidata'])
#    queryentities = []
#    if 'rerankedlists' in json.loads(response[1]):
#        for num,urltuples in json.loads(response[1])['rerankedlists'].iteritems():
#            if json.loads(response[1])['chunktext'][int(num)]['class'] == 'entity':
#                for urltuple in urltuples:
#                    queryentities.append(urltuple[1][0])
#                    break
#    for ent in goldentities:
#        totalentchunks += 1
#        if ent in queryentities:
#            tpentity += 1
#        else:
#            fpentity += 1
#    for ent in queryentities:
#        if ent not in goldentities:
#            fnentity += 1
#    try:
#        precisionentity = tpentity/float(tpentity+fpentity)
#        recallentity = tpentity/float(tpentity+fnentity)
#        f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
#        print("precision entity = ",precisionentity)
#        print("recall entity = ",recallentity)
#        print("f1 entity = ",f1entity)
#    except Exception:
#        pass
    _results.append((response[0],json.loads(response[1])))

#_results = sorted(_results, key=lambda tup: tup[0])

results = []
for result in _results:
    results.append(result)

 
f1 = open('falcon2webqstest.json','w')
print(json.dumps(results),file=f1)
f1.close()
