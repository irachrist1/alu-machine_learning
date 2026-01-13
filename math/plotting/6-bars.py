#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
people = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

width = 0.5
bottom = np.zeros(3)

for i in range(len(fruit)):
    plt.bar(people, fruit[i], width, bottom=bottom, color=colors[i],
            label=fruits[i])
    bottom += fruit[i]

plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.yticks(np.arange(0, 81, 10))
plt.ylim(0, 80)
plt.legend()
plt.show()
