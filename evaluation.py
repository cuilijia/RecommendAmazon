import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# x=[0,1,2,3,4,5,6,7,8,9,10]
x = range(0,11)
y=[527,157,56,18,7,6,0,0,0,0,0]


plt.bar(x, y, label='hit')
plt.legend()

plt.xlabel('number')
plt.ylabel('value')
plt.title(u'Recommended hit hisgram')

plt.show()

sum=0
for i in range(len(y)):
    sum+=y[i]
hitrate=1-y[0]/sum
print(hitrate)

precision=0
for i in range(len(y)):
    precision += x[i]*0.1* y[i]/sum
print(precision)