from typing import List


# 20 Valid Parentheses
# https://leetcode.com/problems/valid-parentheses/
def isValid(s: str) -> bool:
    # using stack, and checking if closing is matching opening
    stack = []
    d = {']': '[', ')': '(', '}': '{'}
    for c in s:
        if c in d.values():
            stack.append(c)
        else:
            open = d.get(c)
            if open is not None:
                if len(stack) == 0:
                    return False
                open_before = stack.pop()
                if open != open_before:
                    return False
    return len(stack) == 0


# 31. Next Permutation (not 100)
# https://leetcode.com/problems/next-permutation/
def nextPermutation(nums: List[int]) -> None:
    # iterating from the end backwards, and then applying some logic about lexicographically ordered permutations
    if not nums or len(nums) <= 1:
        return nums

    found = False
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] < nums[i+1]:
            found = True
            break

    if not found:
        nums.sort()
    else:
        pairs = [(num, i) for i, num in enumerate(nums)]
        filtered = filter(lambda x: x[0]>nums[i], pairs[i+1:])
        val, ind = min(filtered)
        nums[i], nums[ind] = nums[ind], nums[i]
        nums[i+1:] = sorted(list(nums[i+1:]))


# 406 Queue Reconstruction by Height
# https://leetcode.com/problems/queue-reconstruction-by-height/
def reconstructQueue(people: List[List[int]]) -> List[List[int]]:
    # h, k - height, how many people before with larger height
    # preliminary sort input array
    # create resulted array and fill it gradually
    people = sorted(people)
    result = [None] * len(people)
    for h, k in people:
        num_ge = 0
        for i in range(len(result)):
            if num_ge == k and result[i] is None:
                result[i] = [h, k]
                break
            if result[i] is None or result[i][0] == h:
                num_ge += 1
    return result


# 1460. Make Two Arrays Equal by Reversing Sub-arrays (not 100)
# https://leetcode.com/problems/make-two-arrays-equal-by-reversing-sub-arrays/
def canBeEqual(target: List[int], arr: List[int]) -> bool:
    # just sort and compare
    return list(sorted(target)) == list(sorted(arr))
