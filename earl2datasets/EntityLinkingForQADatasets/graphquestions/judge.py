import sys,os,json,re


gold = []
f = open('input/graph.test.entities.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: int(x['question_id']))

for item in d:
    unit = {}
    unit['uid'] = int(item['question_id'])
    unit['question'] = item['utterance']
    unit['entities'] = item['entities']
    gold.append(unit)

f = open('falcon2graphqstest.json')
d1 = json.loads(f.read())

d = sorted(d1, key=lambda x: int(x[0]))

tpentity = 0
fpentity = 0
fnentity = 0
tprelation = 0
fprelation = 0
fnrelation = 0
totalentchunks = 0
totalrelchunks = 0
mrrent = 0
mrrrel = 0
chunkingerror = 0
for queryitem,golditem in zip(d,gold):
    if len(queryitem[1]) == 0:
        continue
    if queryitem[0] != golditem['uid']:
        print(queryitem[0], golditem['uid'])
        print('uid mismatch')
        sys.exit(1)
    queryentities = []
    print(golditem['question'])
    print(set(golditem['entities']))
    for chunk in queryitem[1]['entities']:
        queryentities.append(chunk[0][32:-1])
    print(set(queryentities))
    for goldentity in set(golditem['entities']):
        totalentchunks += 1
        if goldentity in queryentities:
            tpentity += 1
        else:
            fnentity += 1
    for queryentity in set(queryentities):
        if queryentity not in golditem['entities']:
            fpentity += 1

precisionentity = tpentity/float(tpentity+fpentity)
recallentity = tpentity/float(tpentity+fnentity)
f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
print("precision entity = ",precisionentity)
print("recall entity = ",recallentity)
print("f1 entity = ",f1entity)
