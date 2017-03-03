def find_min_idx(a):
    minimum=min(a)
    for i in range(len(a)):
        if a[i]==minimum:
            minimum_idx=i
            
    return minimum_idx

def find_max_idx(a):
    maximum=max(a)
    for i in range(len(a)):
        if a[i]==maximum:
            maximum_idx=i
            
    return maximum_idx