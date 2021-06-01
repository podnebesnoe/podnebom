from typing import List


class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


# 78. Subsets
# https://leetcode.com/problems/subsets/
# https://www.youtube.com/watch?v=LdtQAYdYLcE
def subsets(nums: List[int]) -> List[List[int]]:
    # call recursively, and return list at each call
    subsets_res = [[]]
    for i in range(len(nums)):
        sub_subsets = subsets(nums[i+1:])
        sub_subsets = [[nums[i]] + sub_subset for sub_subset in sub_subsets]
        subsets_res += sub_subsets
    return subsets_res


    '''
    # Kevin solution. call recursively, but add lists into the result instead of returning them
    subsets = []
    def generating(nums, ind, current, subsets):
        subsets.append(current.copy())
        for i in range(ind, len(nums)):
            generating(nums, i + 1, current + [nums[i]], subsets)

    generating(nums, 0, [], subsets)
    return subsets
    '''


    '''
    # uses and returns sets
    result = set()
    result.add(frozenset(nums))
    for i in range(len(nums)):
        l_decreased = [el for ind, el in enumerate(nums) if ind != i]
        result |= subsets_frozenset(l_decreased)
    return result
    '''

# 95. Unique Binary Search Trees II (not 100)
# https://leetcode.com/problems/unique-binary-search-trees-ii/
def generateTrees(n: int) -> List[TreeNode]:
    # call recursively, and return list at each call
    # in recursive call, iterate middle from left to right, and generate lefts and rights
    if n == 0:
        return []

    def helper(left, right):

        if left == right:
            return [None]

        nodes = []
        for i in range(left, right):
            lefts = helper(left, i)
            rights = helper(i+1, right)
            for node_left in lefts:
                for node_right in rights:
                    node = TreeNode(i + 1)
                    node.left = node_left
                    node.right = node_right
                    nodes.append(node)
        return nodes

    return helper(0, n)


# 416. Partition Equal Subset Sum
# https://leetcode.com/problems/partition-equal-subset-sum/
# https://www.youtube.com/watch?v=3N47yKRDed0&
def canPartition(nums: List[int]) -> bool:
    # use recursion (with or without current element)
    # Kevin solution
    total = sum(nums)
    if total%2 == 1:
        return False
    partial_sum = total/2

    def divided(nums, sum, ind):
        if sum == partial_sum:
            return True
        elif sum > partial_sum:
            return False
        if ind >= len(nums):
            return False
        return divided(nums, sum + nums[ind], ind+1) or divided(nums, sum, ind+1)

    return divided(nums, 0, 0)
    '''
    # my solution
    # iterate through array and use recursion. backtracking
    s = sum(nums)
    if s%2:
        return False
    s //= 2

    def canSum(ind_start, target):
        if target == 0:
            return True
        if target < 0:
            return False
        for ind in range(ind_start, len(nums)):
            result = canSum(ind + 1, target - nums[ind])
            if result:
                return True
        return False

    return canSum(0, s)
    '''
