import matplotlib.pyplot as plt
import os

losses = []
log_lr = []
f = open("txts/R18_162256_lrfind_ep5.log")
for id,line in enumerate(f):
    if id % 2 ==0:
        losses.append(float(line[7:].rstrip('\n')))
    else:
        log_lr.append(float(line[8:].rstrip('\n')))
print(min(losses))
print(losses.index(min(losses)))
print(log_lr[losses.index(min(losses))])
plt.plot(log_lr[10:-5],losses[10:-5])
font2 = {'family':'Times New Roman','weight':'normal','size':20}
plt.xlabel("log_lr",font2)
plt.ylabel("loss",font2)
plt.show()