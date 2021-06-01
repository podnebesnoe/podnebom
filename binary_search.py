from typing import List

# 33. Search in Rotated Sorted Array
# https://leetcode.com/problems/search-in-rotated-sorted-array/
def search(nums: List[int], target: int) -> int:
    # first binary search: find shifted start
    # second binary search: standard binary search with applying module for indexation
    left = 0
    right = len(nums) - 1
    length = len(nums)
    if nums[left] > nums[right]:
        while left < right:
            mid = left + (right - left)//2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        start = left
        left = start
        right = start + len(nums) - 1
    while left <= right:
        mid = left + (right - left)//2
        if nums[mid % length] == target:
            return mid % length
        if nums[mid % length] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1


# 34.Find First and Last Position of Element in Sorted Array
# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
def searchRange(nums: List[int], target: int) -> List[int]:
    # first binary search: find ending position
    # second binary search: find starting position
    # difference between first and second steps is regulated by strong / not string inequality
    res = [-1, -1]

    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right - left)//2
        if nums[mid] <= target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        if nums[mid] == target:
            res[1] = mid

    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right - left)//2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] >= target:
            right = mid - 1
        if nums[mid] == target:
            res[0] = mid

    return res


# 162. Find Peak Element
# https://leetcode.com/problems/find-peak-element/
# https://www.youtube.com/watch?v=CFgUQUL7j_c
def findPeakElement(nums: List[int]) -> int:
    # peak can be first or last element as well, so we will get at least one peak by binary search
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = (left + right)//2
        if nums[mid] < nums[mid+1]:
            left = mid + 1
        else:
            right = mid
    return left


# 278. First Bad Version
# https://leetcode.com/problems/first-bad-version/
# https://www.youtube.com/watch?v=SNDE-C86n88

def isBadVersion(version):
    pass

def firstBadVersion(n):
    # standard binary search
    left = 0
    right = n - 1
    while left <= right:
        mid = (left + right) // 2
        if isBadVersion(mid):
            right = mid - 1
        else:
            left = mid + 1
    return left
