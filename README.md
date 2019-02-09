# foundry-hackaton

```
from tools import preprocessing

gen = preprocessing.career_trajectories(10, '../data/HiringPatterns.csv', verbose=True)
encoder = next(gen)
for trj in gen:
    print(trj)
```
