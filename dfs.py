from typing import List

class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


# 17. Letter Combinations of a Phone Number
# https://leetcode.com/problems/letter-combinations-of-a-phone-number/
def letterCombinations(digits: str) -> List[str]:
    # dfs, recursive call with writing results into array

    if digits is None or len(digits) < 1:
        return None

    dmap = {'2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
            '0': ' '}

    cur = digits[0]
    res = []
    combs = letterCombinations(digits[1:])
    for c in dmap[cur]:
        if combs is None:
            res.append(c)
        else:
            for comb in combs:
                res.append(c + comb)
    return res


# 46. Permutations
# https://leetcode.com/problems/permutations/
def permute(nums: List[int]) -> List[List[int]]:
    # recursive call which returns array
    # each recursive call extend this array
    res = []
    for i in range(len(nums)):
        perms = permute(nums[:i] + nums[i+1:])
        if len(perms):
            res += [[nums[i]] + perm for perm in perms]
        else:
            res += [[nums[i]]]
    return res


# 79. Word Search
# https://leetcode.com/problems/word-search/
# https://www.youtube.com/watch?v=vYYNp0Jrdv0
def exist(board: List[List[str]], word: str) -> bool:
    # dfs, use 2D grid with visited cells (which are getting set and unset)

    n = len(board)
    m = len(board[0])

    def dfs(word, board, visited, i, j, n, m):
        if len(word) == 0:
            return True
        if i < 0 or j < 0 or i >= n or j >= m:
            return False
        if word[0] != board[i][j] or visited[i][j]:
            return False
        for i_1, j_1 in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            visited[i][j] = True
            if dfs(word[1:], board, visited, i_1, j_1, n, m):
                return True
            visited[i][j] = False
        return False

    for i in range(n):
        for j in range(m):
            visited = [[False] * len(board[0]) for _ in range(len(board))]
            if dfs(word, board, visited, i, j, n, m):
                return True

    return False


# 130. Surrounded Regions (not 100)
# https://leetcode.com/problems/surrounded-regions/
def solve(board: List[List[str]]) -> None:
    # dfs for border cells. use special character '1' for visited cells
    # after that, postprocess cells with special character
    if not board or not board[0]:
        return None

    def dfs(i, j):
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            return
        if board[i][j] == '1':
            return
        if board[i][j] == 'O':
            board[i][j] = '1'
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)

    for i in range(len(board)):
        dfs(i, 0)
        dfs(i, len(board[0]) - 1)

    for j in range(len(board[0])):
        dfs(0, j)
        dfs(len(board) - 1, j)

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            if board[i][j] == '1':
                board[i][j] = 'O'


# 207. Course Schedule
# https://leetcode.com/problems/course-schedule/
def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    # first step: build a dictionary: key is the course, value is courses which should be taken prior
    # second step: dfs with adding/removing from visited
    d = {}
    for p in prerequisites:
        if p[0] not in d:
            d[p[0]] = []
        d[p[0]].append(p[1])

    visited = set()

    def dfs(k):
        if k in visited:
            return False
        visited.add(k)
        if k in d:
            for v in d[k]:
                if not dfs(v):
                    return False
            d[k] = []
        visited.remove(k)
        return True

    for k in d:
        if not dfs(k):
            return False

    return True


# 200. Number of Islands
# https://leetcode.com/problems/number-of-islands/
# https://www.youtube.com/watch?v=o8S2bO3pmO4
def numIslands(grid: List[List[str]]) -> int:
    # dfs, zero visited cells, increment global variable num

    num = 0
    n = len(grid)
    m = len(grid[0])

    def dfs(i, j, n, m):
        if i<0 or j<0 or i>n-1 or j>m-1 or grid[i][j] == '0':
            return
        grid[i][j] = '0'
        for i_1, j_1 in ((i-1, j), (i+1, j), (i,j-1), (i,j+1)):
            dfs(i_1, j_1, n, m)

    for i in range(n):
        for j in range(m):
            if grid[i][j] == '1':
                dfs(i, j, n, m)
                num += 1

    return num


# 286. Walls and Gates (locked)
# https://leetcode.com/problems/walls-and-gates/
# https://www.youtube.com/watch?v=Pj9378ZsCh4
def walls_and_gates(grid):
    # dfs, recalculate values in 2D array

    if not grid or not grid[0]:
        return

    def dfs(dist, i, j, n, m):
        if i < 0 or j < 0 or i >= n or j >= m or grid[i][j] < dist:
            return
        grid[i][j] = dist
        for i_1, j_1 in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
            dfs(dist + 1, i_1, j_1, n, m)

    n, m = len(grid), len(grid[0])
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 0:
                dfs(0, i, j, n, m)

