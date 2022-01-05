import numpy as np
qt = {
    'a': [1, 0, 1, 1],
    'b': [1, 1, 0, 0]    
}

arr = np.array(list(qt.values()))
print(arr)

dic = {'a': 1, 'b': 5, 'c': 3}
sorted_dic = sorted(dic.items(), key=lambda x: x[1])

print(sorted_dic)