a = [1,3,5,7,8, 5 ]

diff = [j - i for i, j in zip(a, a[1:])]
print(diff)