from typing import List
import heapq
import math


# 215. Kth Largest Element in an Array
# https://leetcode.com/problems/kth-largest-element-in-an-array/
# https://www.youtube.com/watch?v=FrWq2rznPLQ
def findKthLargest(self, nums: List[int], k: int) -> int:
    # solution which works for streams
    # keep in the heap not more than k numbers
    if k > len(nums):
        return None

    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heapq.heappop(heap)
    '''
    # solution with heapify
    heap = [-num for num in nums]
    heapq.heapify(heap)
    for _ in range(k):
        res = heapq.heappop(heap)
    return -res
    '''


# 253 (locked)
# https://leetcode.com/problems/meeting-rooms-ii/
# https://www.youtube.com/watch?v=PWgFnSygweI
def meeting_rooms(intervals):
    # first step: preliminary sort
    # second step: pushing (and maybe popping) into the heap
    if not intervals:
        return 0
    sorted(intervals, key = lambda x: x[0])
    heap = []
    heap.heappush(heap, intervals[0][1])
    for start, end in intervals[1:]:
        earliest_end = heap[0]
        if start > earliest_end:
            heap.heappop(heap)
        heap.heappush(heap, end)

    return len(heap)


# 347. Top K Frequent Elements
# https://leetcode.com/problems/top-k-frequent-elements/
def topKFrequent(nums: List[int], k: int) -> List[int]:
    # first step: build a dictionary
    # second step: use heapify for (-value, key)
    # third step: extract k elements from heap
    d = {}
    for num in nums:
        if num not in d:
            d[num] = 0
        d[num] += 1

    heap = [(-v, k) for k, v in d.items()]
    heapq.heapify(heap)

    result = []
    for _ in range(k):
        if not heap:
            break
        item = heapq.heappop(heap)
        result.append(item[1])

    return result


# 621. Task Scheduler
# https://leetcode.com/problems/task-scheduler/
# https://www.youtube.com/watch?v=ySTQCRya6B0
def leastInterval(tasks: List[str], n: int) -> int:
    # use heap and temporary array to heappop and heappush
    from collections import Counter
    if n == 0:
        return len(tasks)
    result = 0
    counter = Counter(tasks)
    heap = [-val for val in counter.values()]
    heapq.heapify(heap)
    while heap:
        arr = []
        for _ in range(n + 1):
            if heap:
                val = -heapq.heappop(heap)
                arr.append(val - 1)
            else:
                break
        for val in arr:
            if val:
                heapq.heappush(heap, -val)
        if heap:
            result += n + 1
        else:
            result += len(arr)
    return result


# 739. Daily Temperatures
# https://leetcode.com/problems/daily-temperatures/
def dailyTemperatures(T: List[int]) -> List[int]:
    # iterate through array and push/pop from the heap
    # put (val, ind) from enumerate into a heap
    if not T:
        return None
    if len(T) == 1:
        return 0

    res = [None] * len(T)

    heap = []

    for ind, num in enumerate(T):
        while heap:
            if heap[0][0] < num:
                num_prev, ind_prev = heapq.heappop(heap)
                res[ind_prev] = ind - ind_prev
            else:
                break
        heapq.heappush(heap, (num, ind))

    while heap:
        num_prev, ind_prev = heapq.heappop(heap)
        res[ind_prev] = 0

    return res


# 973. K Closest Points to Origin
# https://leetcode.com/problems/k-closest-points-to-origin/
# https://www.youtube.com/watch?v=1rEUgAG7f_c
def kClosest(points: List[List[int]], k: int) -> List[List[int]]:
    # convert elements to arrays and use 0th element of array for comparison in order to utilize heap
    points = [(x**2 + y**2, x, y) for x, y in points]
    heapq.heapify(points)
    res = []
    for i in range(k):
        if not points:
            break
        dist, x, y = heapq.heappop(points)
        res.append([x, y])

    return res


# 1046. Last Stone Weight
# https://leetcode.com/problems/last-stone-weight/
# https://www.youtube.com/watch?v=fBPS7PtPtaE
def lastStoneWeight(stones: List[int]) -> int:
    # heapify array and keep heappoping
    stones = [-i for i in stones]
    heapq.heapify(stones)
    while len(stones) > 1:
        a, b = heapq.heappop(stones), heapq.heappop(stones)
        diff = abs(a - b)
        if diff > 0:
            heapq.heappush(stones, -diff)
    return -stones[0] if stones else 0


# 1167 (locked)
# https://leetcode.com/problems/minimum-cost-to-connect-sticks/
# https://www.youtube.com/watch?v=3dqR2nYElyw
def combine_sticks(nums):
    # heapify array and keep heappoping
    heapq.heapify(nums)
    cost = 0
    while len(nums) > 1:
        a, b = heapq.heappop(nums), heapq.heappop(nums)
        sum = a + b
        cost += sum
        heapq.heappush(nums, sum)
    return cost
