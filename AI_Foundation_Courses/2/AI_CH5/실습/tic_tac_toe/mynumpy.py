# mynumpy.py

def argmax(items):
    return items.index(max(items))

def argmin(items):
    return items.index(min(items))

if __name__ == '__main__':
    nums = [1, 534, 123, -123, 34]
    print argmax(nums)
    print argmin(nums)
