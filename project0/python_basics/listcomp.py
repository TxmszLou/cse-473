nums = [1,2,3,4,5,6]
oddNums = [x for x in nums if x % 2 == 1]
print(oddNums)
oddNumsPlusOne = [x+1 for x in nums if x % 2 ==1]
print(oddNumsPlusOne)


# exercise
def lowerIt (lst):  return [str.lower() for str in lst if len(str) > 5]

lowerItL = lambda lst : [str.lower() for str in lst if len(str) > 5]
