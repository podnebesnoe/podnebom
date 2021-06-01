from typing import List


# 11.Container With Most Water
# https://leetcode.com/problems/container-with-most-water/
def maxArea(height: List[int]) -> int:
    # iterators from start and end
    # update answer which is maximal value
    if len(height) <= 1:
        return 0
    ind1 = 0
    ind2 = len(height) - 1
    def get_s(ind1, ind2):
        return min(height[ind1], height[ind2]) * (ind2 - ind1)
    s = 0
    while ind1 < ind2:
        s_cur = get_s(ind1, ind2)
        if s_cur > s:
            s = s_cur
        if height[ind1] < height[ind2]:
            ind1 += 1
        else:
            ind2 -= 1
    return s


# 15. 3Sum
# https://leetcode.com/problems/3sum/
def threeSum(nums: List[int]) -> List[List[int]]:
    # preliminary sort
    # iterator from beginning and end, while beginning is also iterating
    nums = sorted(nums)
    res = []
    for ind in range(len(nums) - 2):
        num = nums[ind]
        if ind > 0 and nums[ind] == nums[ind - 1]: # can be modified and moved to the end of the for block, in order to extend ot for fourSum
            continue
        ind1 = ind + 1
        ind2 = len(nums) - 1
        while ind1 < ind2:
            sum = num + nums[ind1] + nums[ind2]
            if sum == 0:
                res.append([num, nums[ind1], nums[ind2]])
                while ind1 < ind2 and nums[ind1] == nums[ind1 + 1]:
                    ind1 += 1
                while ind1 < ind2 and nums[ind2] == nums[ind2 - 1]:
                    ind2 -= 1
                ind1 += 1
                ind2 -= 1
            elif sum < 0:
                ind1 += 1
            else:
                ind2 -= 1
    return res

# 16. 3Sum Closest
# https://leetcode.com/problems/3sum-closest/
def threeSumClosest(nums: List[int], target: int) -> int:
    # solution is similar to above
    # update the minimal answer
    nums = sorted(nums)
    res = None
    for ind in range(len(nums) - 2):
        ind1 = ind + 1
        ind2 = len(nums) - 1
        while ind1 < ind2:
            sum = nums[ind] + nums[ind1] + nums[ind2]
            if res is not None:
                if abs(sum - target) < abs(res - target):
                    res = sum
            else:
                res = sum
            if sum < target:
                ind1 += 1
            else:
                ind2 -= 1
    return res


# 18. 4Sum
# https://leetcode.com/problems/4sum/
def fourSum(nums: List[int], target: int) -> List[List[int]]:
    # extension of the threeSum with extra loop
    # can be extended recursively to kSum
    pass


# 26. Remove Duplicates from Sorted Array
# https://leetcode.com/problems/remove-duplicates-from-sorted-array/
def removeDuplicates(nums: List[int]) -> int:
    # two iterators
    cur = -1
    for i in range(len(nums)):
        if cur < 0 or nums[i] != nums[cur]:
            cur = cur + 1
            nums[cur] = nums[i]
    return cur + 1


# 53. Maximum Subarray
# https://leetcode.com/problems/maximum-subarray/
def maxSubArray(nums: List[int]) -> int:
    # iterate and update temporary answer and final answer
    maxEndingHere = maxSofFar = nums[0]
    for i in range(1, len(nums)):
        maxEndingHere = max(maxEndingHere + nums[i], nums[i])
        maxSofFar = max(maxEndingHere, maxSofFar)
    return maxSofFar


# 56. Merge Intervals
# https://leetcode.com/problems/merge-intervals/
def merge(intervals: List[List[int]]) -> List[List[int]]:
    # preliminary sort by start
    # iterate and start new interval is new start is after maximal previous end
    if len(intervals) < 1:
        return intervals
    intervals = sorted(intervals, key = lambda x: x[0])
    res = []
    start = intervals[0][0]
    end = intervals[0][1]
    for interval in intervals[1:]:
        if interval[0] > end:
            res.append([start, end])
            start = interval[0]
            end = interval[1]
        end = max(end, interval[1])
    res.append([start, end])
    return res


# 57. Insert Interval (not 100)
# https://leetcode.com/problems/insert-interval/
def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    # looking for intervals (in between) affected by newInterval, and then merging these intervals (in between) with newInterval,
    # and appending intervals at the left and at the right

    if not intervals:
        return [newInterval]

    if intervals[0][0] > newInterval[1]:
        return [newInterval] + intervals
    if intervals[-1][1] < newInterval[0]:
        return intervals + [newInterval]

    s = newInterval[0]
    e = newInterval[1]
    for ind, interval in enumerate(intervals):
        if s <= interval[1]:
            break
    si = ind
    for ind, interval in enumerate(intervals):
        if e < interval[0]:
            break
    ei = ind
    if e >= intervals[ei][0]:
        ei += 1

    midintervals = sorted(intervals[si:ei] + [newInterval], key=lambda x: x[0])
    res = []
    curs = None
    cure = None
    for s, e in midintervals:
        if curs is None:
            curs = s
            cure = e
        else:
            if s > cure:
                res.append(curs, cure)
                curs = s
                cure = e
            else:
                cure = max(cure, e)
    res.append([curs, cure])

    return intervals[:si] + res + intervals[ei:]


# 75. Sort Colors
# https://leetcode.com/problems/sort-colors/
def sortColors(nums: List[int]) -> None:
    # iterators from the beginning and the end, and extra iterator from beginning moving faster than the first iterator
    ind0 = 0
    ind2 = len(nums) - 1
    ind = 0
    while ind0 < ind2 and ind <= ind2:
        if nums[ind] == 0:
            nums[ind] = nums[ind0]
            nums[ind0] = 0
            ind += 1
            ind0 += 1
        elif nums[ind] == 2:
            nums[ind] = nums[ind2]
            nums[ind2] = 2
            ind2 -= 1
        else:
            ind += 1


# 88. Merge Sorted Array (not 100)
# https://leetcode.com/problems/merge-sorted-array/
def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    # iterator from each array, use empty space from one array for filling final results

    if not nums1 or not nums2:
        return
    i = n + m - 1
    i1 = m - 1
    i2 = n - 1
    while i >= 0:
        if i1 >= 0 and (i2 < 0 or nums1[i1] > nums2[i2]):
            nums1[i] = nums1[i1]
            i1 -= 1
        else:
            nums1[i] = nums2[i2]
            i2 -= 1
        i -= 1


# 121. Best Time to Buy and Sell Stock
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# https://www.youtube.com/watch?v=mj7N8pLCJ6w
def maxProfit(prices: List[int]) -> int:
    # iterate and update minimum value and answer
    if len(prices) < 2:
        return 0
    minval = prices[0]
    profit = 0
    for price in prices[1:]:
        profit = max(price - minval, profit)
        minval = min(price, minval)
    return profit


# 152. Maximum Product Subarray
# https://leetcode.com/problems/maximum-product-subarray/
def maxProduct(nums: List[int]) -> int:
    # iterate and update temporary subanswers and answer
    answer = None
    curMin = curMax = 1
    for num in nums:
        tmpCurMax = max(curMax * num, curMin * num, num)
        curMin = min(curMax * num, curMin * num, num)
        curMax = tmpCurMax

        answer = max(curMax, answer) if answer is not None else curMax

    return answer


# 283. Move Zeroes
# https://leetcode.com/problems/move-zeroes/
# https://www.youtube.com/watch?v=1PEncepEIoE
def moveZeroes(nums: List[int]) -> None:
    # two iterators moving with different speed
    ind2 = 0
    for ind1 in range(len(nums)):
        if nums[ind1] != 0:
            nums[ind2] = nums[ind1]
            ind2 += 1

    while ind2 < len(nums):
        nums[ind2] = 0
        ind2 += 1


# 209. Minimum Size Subarray Sum (not 100)
# https://leetcode.com/problems/minimum-size-subarray-sum/
def minSubArrayLen(target: int, nums: List[int]) -> int:
    # 2 sliding window iterators to support sum not less than target
    if not nums:
        return 0
    i1 = 0
    i2 = 0
    cursum = nums[0]
    res = len(nums) + 1
    while i2 < len(nums):
        if cursum >= target:
            return 1
        while cursum < target and i2 < len(nums) - 1:
            i2 += 1
            cursum += nums[i2]
        if cursum < target:
            break
        while cursum >= target:
            res = min(res, i2 - i1 + 1)
            if i1 == i2:
                return 1
            cursum -= nums[i1]
            i1 += 1

    return res if res <= len(nums) else 0


# 334. Increasing Triplet Subsequence (not 100)
# https://leetcode.com/problems/increasing-triplet-subsequence/
def increasingTriplet(nums: List[int]) -> bool:
    # iterating through array and updating first and second intermediate subanswer

    if not nums or len(nums) < 3:
        return False

    first = None
    second = None

    for i in range(len(nums)):

        if second is not None and nums[i] > second:
            return True

        if first is not None and nums[i] > first:
            if second is None or nums[i] < second:
                second = nums[i]

        if first is None or nums[i] < first:
            first = nums[i]

    return False


# 581. Shortest Unsorted Continuous Subarray
# https://leetcode.com/problems/shortest-unsorted-continuous-subarray/
def findUnsortedSubarray(nums: List[int]) -> int:
    # iterators from the beginning and the end
    # after that, iterators outwards
    if not nums:
        return 0
    if len(nums) == 1:
        return 0
    i1 = 0
    i2 = len(nums) - 1
    while True:
        if nums[i1] > nums[i2]:
            break
        if nums[i1] <= nums[i1 + 1] and nums[i1 + 1] <= nums[i2]:
            i1 += 1
        elif nums[i2 - 1] <= nums[i2] and nums[i2 - 1] >= nums[i1]:
            i2 -= 1
        else:
            break
        if i1 >= i2:
            return 0

    minval = min(nums[i1:i2 + 1])
    maxval = max(nums[i1:i2 + 1])

    while i1 >= 0 and minval < nums[i1]:
        i1 -=1

    while i2 < len(nums) and maxval > nums[i2]:
        i2 +=1

    return i2 - i1 - 1
