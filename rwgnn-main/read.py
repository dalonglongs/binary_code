import pickle
from collections import defaultdict
import json
import numpy as np
import time
import matplotlib.pyplot as plt

name = {'osdist_acc', 'osdist_time', 'cosdist_acc', 'cosdist_time', 'sjdist_acc', 'sjdist_time'}
duliang = defaultdict(list)


for key in name:
    f = open(key+'.pkl', 'rb')
    a = 0
    for i in range(15):
        if "time" in key:
            text = float(pickle.load(f))
            a = a + text
            duliang[key].append(float(a))
        else:
            text = pickle.load(f)
            duliang[key].append(float(text))
epoch = []
for i in range(15):
    epoch.append(i + 1)
# f = json.dumps(duliang, indent=4)
# filename = 'all_record.json'
# with open(filename, 'w') as file_obj:
#     file_obj.write(f)
print(duliang['osdist_time'])
print(duliang['cosdist_time'])
print(duliang['sjdist_time'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(epoch, duliang['osdist_acc'], label='euclidean metric')
ax.plot(epoch, duliang['cosdist_acc'], color='red', label='cosine similarity')
ax.plot(epoch, duliang['sjdist_acc'], color='yellow', label='deep non-linear metric')
plt.legend()
ax.set_xlabel("epochs")
ax.set_ylabel("accury")
plt.show()

