import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os

#new_ticks = np.linspace(0, 70, 7)
#plt.xticks(new_ticks)

dir_name = "nzb"
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)


x_data = []
y1_data = []
y2_data = []




f = open('E:/wangchen/data/voltage/nzb/nzb1.csv', 'r', encoding='ANSI')
f1 = csv.reader(f)


n = 0
for info in f1:
    n=n+1
    y1_data.append(float(info[0]))
    y2_data.append(float(info[1]))

k=20
m=int(n/k)

x = []
y1 = []
y2 = []

for i in range(0,m):

    x.clear()
    y1.clear()
    y2.clear()

    for j in range(1, k+1):
        w = i * k + j - 1
        x.append(j)
        y1.append(y1_data[w])
        y2.append(y2_data[w])



    plt.plot(x, y1, c= 'red',  linestyle='-',  linewidth=1, label="1")
    plt.plot(x, y2, c= 'blue',  linestyle='-',  linewidth=1, label="2")



    plt.savefig(dir_name +"/"+str(i)+".png",dpi=300)
    plt.close()















