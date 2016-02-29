import string
label_stats = {}

with open('Data/pnp-validate.txt') as datafile:
    count = 0
    for line in datafile:
        count += 1
        label = line.split('\t')[0]
        sample = line.split('\t')[1]
        words = sample.split(' ')

        if label not in label_stats:
            label_stats[label] = 0
        # if ' ' in sample:
        for i in range(len(sample)):
            if sample[i] in '1234567890':
                label_stats[label] += 1

print count
print label_stats