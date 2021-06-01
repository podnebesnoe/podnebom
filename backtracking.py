from typing import List


# 22. Generate Parentheses
# https://leetcode.com/problems/generate-parentheses/
def generateParenthesis(n: int) -> List[str]:
    '''
    # build dpa elements of which are combinations
    if n == 0:
        return None
    res = [None] * n
    res[0] = ['()']
    for i in range(1, n):
        s = set()
        for item in res[i-1]:
            s.add('(%s)' % item)
        for j in range(i):
            for item1 in res[j]:
                for item2 in res[i - j - 1]:
                    s.add(item1 + item2)
        res[i] = list(s)

    return res[-1]
    '''
    # storing result into external array
    # recursive function with increasing open and close counter
    # in the final basic case, this function is just adding string into result
    res = []
    def bt(cur, open, close):
        if len(cur) == 2 * n:
           res.append(cur)
        else:
            if open < n:
                bt(cur + '(', open + 1, close)
            if close < open:
                bt(cur + ')', open, close + 1)

    bt('', 0, 0)
    return res


# 39 Combination Sum
# https://leetcode.com/problems/combination-sum/
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    # storing result into external array
    # recursive function with decreasing target
    # in the case of success, this function is adding combination into result
    result = []

    def get_combinations(candidates, ind, target, cur, result):
        if target == 0: # basic case
            result.append(cur.copy())
            return
        for i in range(ind, len(candidates)):
            if candidates[i] <= target:
                cur.append(candidates[i])
                get_combinations(candidates, i, target - candidates[i], cur, result)
                cur.pop()

    get_combinations(candidates, 0, target, [], result)

    return result


# 40. Combination Sum II
# https://leetcode.com/problems/combination-sum-ii/
# https://www.youtube.com/watch?v=IER1ducXujU
def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
    # similar to the solution above
    # sort preliminary and skip repeating numbers
    candidates = sorted(candidates)

    result = []

    def get_combinations(ind, cur, target, result):
        if target == 0:
            result.append(cur.copy())
        i = ind + 1
        while i < len(candidates):
            if candidates[i] <= target:
                cur.append(candidates[i])
                get_combinations(i, cur, target - candidates[i], result)
                cur.pop()
            i += 1
            while i < len(candidates) and candidates[i] == candidates[i - 1]:
                i += 1

    get_combinations(-1, [], target, result)

    return result


# 97. Interleaving String (not 100)
# https://leetcode.com/problems/interleaving-string/
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    # backtracking with checking condition in the basic case
    if s1 is None or s2 is None or s3 is None:
        return False

    if len(s3) != len(s1) + len(s2):
        return False

    def helper(i1, i2, i3, flag):
        if i3 == len(s3):
            return True
        if flag and i1 < len(s1):
            while i1 < len(s1) and s3[i3] == s1[i1]:
                if helper(i1+1, i2, i3+1, False):
                    return True
                i1 += 1
                i3 += 1
        else:
            while i2 < len(s2) and s3[i3] == s2[i2]:
                if helper(i1, i2+1, i3+1, True):
                    return True
                i2 += 1
                i3 += 1
        return False

    return helper(0, 0, 0, True) or helper(0, 0, 0, False)
