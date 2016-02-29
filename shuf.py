import random
with open('Data/pnp-train.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('Data/pnp-train2.txt','w') as target:
    for _, line in data:
        target.write( line )