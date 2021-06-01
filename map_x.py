from typing import List

# 1. Two Sum
# https://leetcode.com/problems/two-sum/
# https://www.youtube.com/watch?v=Aql6zHkONek
def twoSum(nums: List[int], target: int) -> List[int]:
    # just use dictionary
    d = {}
    for ind, num in enumerate(nums):
        another = target - num
        if another in d:
            return [d[another], ind]
        else:
            d[num] = ind
    return [None, None]


# 210. Course Schedule II (not 100)
# https://leetcode.com/problems/course-schedule-ii/
def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    # use dictionary for prerequisite dependencies
    # loop: find courses which don't have dependencies and remove them from dependencies of other courses

    d = {k: set() for k in range(numCourses)}
    for p in prerequisites:
        d[p[0]].add(p[1])

    res = []

    while len(res) < numCourses:
        cur = set()
        for k, v in d.items():
            if not v:
                cur.add(k)

        if not cur:
            return []

        for k in d:
            d[k] = d[k].difference(cur)

        for k in cur:
            del d[k]

        res.extend(list(cur))

    return res


# 448. Find All Numbers Disappeared in an Array
# https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/
# https://www.youtube.com/watch?v=efU_3Da3DV0
def findDisappearedNumbers(nums: List[int]) -> List[int]:
    # use set
    s = set()
    for num in nums:
        s.add(num)
    return list(set(1, range(len(nums)) + 1).difference(s))


# 530. Subarray Sum Equals K
# https://leetcode.com/problems/subarray-sum-equals-k/
def subarraySum(nums: List[int], k: int) -> int:
    # dictionary of cumulative sums
    # difference between cumulative sums is sum of continuous subarray
    if not nums:
        return 0

    res = 0
    sum = 0
    sums = dict()

    for num in nums:
        if sum not in sums:
            sums[sum] = 0
        sums[sum] += 1
        sum += num
        if sum - k in sums:
            res += sums[sum - k]

    return res


# 763. Partition Labels
# https://leetcode.com/problems/partition-labels/
def partitionLabels(S: str) -> List[int]:
    # build preliminary dictionary: key is character, value is (start, end), sort its values and iterate through them:
    # start new partition if new start is after maximal previous end
    d = {}
    for ind, ch in enumerate(S):
        if ch not in d:
            d[ch] = [ind, ind]
        else:
            d[ch][1] = ind

    arr = d.values()
    arr = sorted(arr)

    res = []
    cur_start = -1
    cur_end = -1
    for start, end in arr:
        if start > cur_end:
            if cur_start >= 0:
                res.append(cur_end - cur_start + 1)
            cur_start = start
            cur_end = end
        else:
            cur_end = max(cur_end, end)

    res.append(cur_end - cur_start + 1)

    return res


# 771. Jewels and Stones
# https://leetcode.com/problems/jewels-and-stones/
# https://www.youtube.com/watch?v=9Reqqk60Nv4
def numJewelsInStones(jewels: str, stones: str) -> int:
    # use set
    jewels = set(jewels)
    return len([i for i in stones if i in jewels])


# 904. Fruit Into Baskets
# https://leetcode.com/problems/fruit-into-baskets/
# https://www.youtube.com/watch?v=za2YuucS0tw
def totalFruit(tree: List[int]) -> int:
    # iterate and update dictionary where value is the latest index of entrance
    # if size of dictionary is more than 2, then remove earliest entering element
    # update the answer every iteration
    # the same as longest substring with the same 2 characters
    if not tree:
        return 0
    result = 1
    d = {}
    i = 0
    start = 0
    while i < len(tree):
        if len(d) <= 2:
            d[tree[i]] = i
            i += 1
        if len(d) > 2:
            min_v = len(tree)
            for v in d.values():
                min_v = min(min_v, v)
            del d[tree[min_v]]
            start = min_v + 1
        result = max(result, i-start)

    return result
