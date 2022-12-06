import numpy as np
import matplotlib.pyplot as plt



input = [
    [1, 2, 3, 5, 6, 7, 8],
    [-1, 4, 2, 2, 3, 4, 5],
    [1, 2, 3, 1, 23, 4, 1],
    [2, 3, 1, 3, 4, 2, 1]
]

def maxpool(input:list, kernel:tuple, stride:tuple):
    output = []
    k = 0
    while kernel[0]+k <= len(input): # stride column
        s = 0
        row = []
        while kernel[1]+s <= len(input[0]): # stride row
            window = set()
            for i in range(0, kernel[0]): # kernel window
                window = window.union(set(input[i+k][s:kernel[1]+s]))
            max_value = max(window)
            row.append(max_value)
            s += stride[0]
        k += stride[1]
        output.append(row) 
    return output

def nmaxpool(input:list, kernel:tuple, stride:tuple):
    # stride cols
    k = 0
    output = []
    while kernel[0]+k <= len(input):
        s = 0
        row = []
        while kernel[1]+s <= len(input[0]):
            window = set()
            for i in range(0, kernel[0]):
                window = window.union(set(input[i+k][s:kernel[1]+s]))
            s += stride[1]
            max_value = max(window)
            row.append(max_value)
        output.append(row)
        k += stride[0]
    return output


image = maxpool(input, (2,2), (1,1))
image2 = nmaxpool(input, (2,2), (1,1))

f, ax = plt.subplots(1,3)
ax[0].imshow(input)
ax[1].imshow(image)
ax[2].imshow(image2)
plt.show()



