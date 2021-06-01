from typing import List


# 134. Gas Station (not 100)
# https://leetcode.com/problems/gas-station/
def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    # iterate through array twice and keep best local solution
    # if local solution fails, reset it to another local solution
    # iterate until actual solution will be found, otherwise no solution
    n = len(gas)
    tank = 0
    maxlen = 0
    for i in range(2 * n):
        tank += gas[i%n] - cost[i%n]
        if tank >= 0:
            maxlen += 1
            if maxlen >= n:
                return i - (maxlen - 1)
        else:
            tank = 0
            maxlen = 0
    return -1

