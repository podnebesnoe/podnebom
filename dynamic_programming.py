from typing import List


# 45. Jump Game II
# https://leetcode.com/problems/jump-game-ii/
def jump(nums: List[int]) -> int:
    '''
    # dfs
    if len(nums) <= 1:
        return 0
    jumps_min = None
    jump_max = nums[0]
    for j in range(jump_max, 0, -1):
        jumps_cur = jump(nums[j:])
        if jumps_cur is not None:
            jumps_cur += 1
            if jumps_min is None:
                jumps_min = jumps_cur
            elif jumps_cur < jumps_min:
                jumps_min = jumps_cur
    return jumps_min
    '''

    # build dpa

    dpa = [0] + [None] * (len(nums) - 1)

    for i in range(1, len(nums)):
        for j in range(0, i):
            if nums[j] + j >= i:
                if dpa[i] is None:
                    dpa[i] = dpa[j] + 1
                else:
                    dpa[i] = min(dpa[i], dpa[j] + 1)
    return dpa[-1]


# 55. Jump Game
# https://leetcode.com/problems/jump-game/
def canJump(nums: List[int]) -> bool:
    # build dpa
    if not nums:
        return False
    res = [False] * len(nums)
    res[0] = True
    for i in range(1, len(nums)):
        for j in range(i-1, -1, -1):
            if res[j] and nums[j] >= i - j:
                res[i] = True
                break

    return res[-1]


# 62. Unique Paths
# https://leetcode.com/problems/unique-paths/
def uniquePaths(m: int, n: int) -> int:
    # build 2D dpa
    if m < 1 or n < 1:
        return 0
    space = [[0] * n] * m
    space[0][0] = 1
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            way1 = way2 = 0
            if i > 0:
                way1 = space[i - 1][j]
            if j > 0:
                way2 = space[i][j - 1]

            space[i][j] = way1 + way2

    return space[m - 1][n - 1]


# 63. Unique Paths II (not 100)
def uniquePathsWithObstacles(obstacleGrid: List[List[int]]) -> int:
    # build 2D dpa
    if not obstacleGrid:
        return 0
    if not obstacleGrid[0]:
        return 0

    if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1:
        return 0

    for i in range(len(obstacleGrid)):
        for j in range(len(obstacleGrid[0])):
            if i==0 and j==0:
                obstacleGrid[0][0] = 1
                continue
            if obstacleGrid[i][j] == 1:
                obstacleGrid[i][j] = 0
                continue

            val1 = obstacleGrid[i-1][j] if i > 0 else 0
            val2 = obstacleGrid[i][j-1] if j > 0 else 0
            obstacleGrid[i][j] = val1 + val2

    return obstacleGrid[-1][-1]


# 64. Minimum Path Sum
# https://leetcode.com/problems/minimum-path-sum/
def minPathSum(grid: List[List[int]]) -> int:
    # build 2D dpa (actually, we can reuse given array)
    m = len(grid)
    if m < 1:
        return 0
    n = len(grid[0])
    if n < 1:
        return 0
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            vals = []
            if i > 0:
                vals.append(grid[i - 1][j])
            if j > 0:
                vals.append(grid[i][j - 1])
            grid[i][j] = grid[i][j] + min(vals)

    return grid[-1][-1]


# 70. Climbing Stairs
# https://leetcode.com/problems/climbing-stairs/
# https://www.youtube.com/watch?v=uHAToNgAPaM
def climbStairs(n: int) -> int:
    # build dpa
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 2

    dp = [None] * n
    dp[0] = 1
    dp[1] = 2

    for i in range(n - 2):
        dp[i + 2] = dp[i] + dp[i + 1]

    return dp[-1]


# 91. Decode Ways
# https://leetcode.com/problems/decode-ways/
# https://www.youtube.com/watch?v=cQX3yHS0cLo
def numDecodings(s: str) -> int:
    '''
    # top to bottom dynamic programming
    arr = [None] * len(word)
    arr[0] = 1

    def num_ways_dynamic(word, k):
        if arr[k] is not None:
            return arr[k]
        result = num_ways_dynamic(word, k-1)
        if k >= 1 and int(word[k-1:k+1]) <= 26:
            if k >= 2:
                result += num_ways_dynamic(word, k - 2)
            else:
                result += 1
        arr[k] = result
        return result

    return num_ways_dynamic(word, len(word)-1)
    '''
    # build dpa
    if not s:
        return 0
    res = [0] * len(s)
    res[0] = 1 if int(s[0]) >=1 else 0
    for i in range(1, len(s)):
        if int(s[i-1]) == 1 or (int(s[i-1]) == 2 and int(s[i]) <= 6):
            res[i] += res[i - 2] if i >= 2 else 1
        if int(s[i]) >= 1:
            res[i] += res[i-1]

    return res[-1]


# 139. Word Break
# https://leetcode.com/problems/word-break/
def wordBreak(s: str, wordDict: List[str]) -> bool:
    # * l e e t c o d e
    # dfs
    #if len(s) == 0:
    #    return True
    #for i in range(len(s)):
    #    if s[:i+1] in wordDict and wordBreak(s[i+1:], wordDict):
    #        return True
    #return False

    # build dpa with extra element at the beginning

    dpa = [True] + [False] * len(s)

    for i in range(1, len(s) + 1):
        for j in range(i, 0, -1):
            if dpa[j-1] and s[j-1:i] in wordDict:
                dpa[i] = True
                break

    return dpa[len(s)]


# 198. House Robber
# https://leetcode.com/problems/house-robber/
# https://www.youtube.com/watch?v=xlvhyfcoQa4
def rob(nums: List[int]) -> int:
    # build dpa
    if len(nums) < 1:
        return 0
    if len(nums) < 2:
        return nums[0]
    dp = [None] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[1], nums[0])
    for i in range(2, len(nums)):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[-1]


# 221. Maximal Square
# https://leetcode.com/problems/maximal-square/
def maximalSquare(matrix: List[List[str]]) -> int:
    # dpa in 2D: cell is calculated from its surroundings
    # surroundings are extending

    n = len(matrix)
    m = len(matrix[0])

    s = 0
    while s < n and s < m:
        found = False
        for i in range(n - s):
            for j in range(m - s):
                if s > 0:
                    if matrix[i + 1][j] == '0' or matrix[i][j + 1] == '0' or matrix[i + 1][j + 1] == '0':
                        matrix[i][j] = '0'
                if matrix[i][j] == '1':
                    found = True
        if not found:
            break
        s += 1

    return s ** 2


# 256. Paint House (locked)
# https://leetcode.com/problems/paint-house/
# https://www.youtube.com/watch?v=fZIsEPhSBgM
def paint_house(costs):
    # build dpa with arrays as elements. its kinda 2D dpa
    if not costs:
        return 0

    for i in range(1, len(costs)):
        for j in range(3):
            costs[i][j] += min(costs[i-1][(j+1)%3], costs[i-1][(j+2)%3])

    return min(costs[len(costs)-1])


# 279. Perfect Squares
# https://leetcode.com/problems/perfect-squares/
def numSquares(n: int) -> int:
    # build dpa. each next element is potentially using values from all the previous elements
    if n == 0:
        return 0
    sqs = [j ** 2 for j in range(1, n)]
    res = [n] * (n + 1)
    res[0] = 0
    res[1] = 1
    for i in range(2, n + 1):
        for sq in sqs:
            dif = i - sq
            if dif < 0:
                break
            else:
                res[i] = min(res[i], res[dif] + 1)
    return res[-1]


# 300. Longest Increasing Subsequence
# https://leetcode.com/problems/longest-increasing-subsequence/
def lengthOfLIS(nums: List[int]) -> int:
    # build dpa. each next element is potentially using values from all the previous elements
    if not nums:
        return 0
    lts = [0] * len(nums)
    lts[0] = 1

    for i in range(1, len(nums)):
        lts[i] = max([lts[j] + 1 for j in range(0, i) if nums[j] < nums[i]] + [1])

    return max(lts)


# 322. Coin Change
# https://leetcode.com/problems/coin-change/
# https://www.youtube.com/watch?v=1R0_7HqNaW0
def coinChange(coins: List[int], amount: int) -> int:
    # build dpa. each next element is potentially using values from all the previous elements
    da = [0] + [amount + 1] * amount

    for i in range(amount):
        for coin in coins:
            if i + 1 - coin >= 0:
                da[i + 1] = min(da[i + 1], da[i + 1 - coin] + 1)

    return da[amount] if da[amount] < amount + 1 else -1


# 494. Target Sum
# https://leetcode.com/problems/target-sum/
def findTargetSumWays(nums: List[int], S: int) -> int:
    # build dpa for each of the intermediate answers (not for each step/element) and update it on every step

    max_sum = 1000
    if S > max_sum:
        return 0
    dp = [0] * (2 * max_sum + 1)
    dp[max_sum] = 1

    for num in nums:
        dp2 = [0] * (2 * max_sum + 1)
        for i in range(2 * max_sum + 1):
            if dp[i]:
                dp2[i + num] += dp[i]
                dp2[i - num] += dp[i]
        dp = dp2

    return dp[max_sum + S]
