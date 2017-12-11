# python3

import sys

n, m = map(int, sys.stdin.readline().split())
lines = list(map(int, sys.stdin.readline().split()))
rank = [1] * n
parent = list(range(0, n))
ans = max(lines)

def getParent(table):
    if table != parent[table]:
        parent[table] = getParent(parent[table])
    return parent[table]

def merge(destination, source):
    global ans,lines

    realDestination, realSource = getParent(destination), getParent(source)

    if realDestination == realSource:
        return False

    # merge two components
    # use union by rank heuristic
    # update ans with the new maximum table size
    if rank[realSource] > rank[realDestination]:
        lines[realSource] += lines[realDestination]
        if lines[realSource] > ans:
            ans = lines[realSource]
        lines[realDestination] = 0
        parent[realDestination] = realSource
    else:
        lines[realDestination] += lines[realSource]
        if lines[realDestination] > ans:
            ans = lines[realDestination]
        lines[realSource] = 0
        parent[realSource] = realDestination
        if rank[realSource] == rank[realDestination]:
            rank[realDestination] += 1

    return True

for i in range(m):
    destination, source = map(int, sys.stdin.readline().split())
    merge(destination - 1, source - 1)
print(ans)
