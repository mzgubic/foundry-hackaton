import random
from random import choices
import json
n_lvl1 = 10
n_lvl2 = 3
n_lvl3 = 10
n_paths = 10000

random.seed(42)
probs_1 = []
for _ in range(n_lvl1):
    dist = [random.random() for _ in range(n_lvl2)]
    z = sum(dist)
    probs_1.append([p/z for p in dist])
probs_2 = []
for _ in range(n_lvl1):
    probs_2.append([])
    for _ in range(n_lvl2):
        dist = [random.random() for _ in range(n_lvl3)]
        z = sum(dist)
        probs_2[-1].append([p/z for p in dist])
paths = []
for _ in range(n_paths):
    l1 = random.randint(1,n_lvl1)
    l2 = choices([n_lvl1+i+1 for i in range(n_lvl2)],probs_1[l1-1])[0] 
    l3 = choices([n_lvl1+n_lvl2+i+1 for i in range(n_lvl3)],probs_2[l1-1][l2-n_lvl1-1])[0]
    path = [l1,l2,l3]
    if random.randint(0,2)==0:
        path.append(n_lvl1+n_lvl2+n_lvl3+1)

    paths.append(path)
data = {"paths":paths,"probs_1":probs_1,"probs_2":probs_2}
json.dump(data,open("data.json","w"))

