from typing import List


# 7. Reverse Integer
# https://leetcode.com/problems/reverse-integer/
def reverse(x: int) -> int:
    # calculation fraction and reminder, and use multiplication on 10 for the future answer
    sign = 1 if x > 0 else -1
    x = x * sign
    res = 0
    while x:
        digit = x % 10
        x = int(x / 10)
        res = res * 10 + digit
    return res * sign


# 9. Palindrome Number
# https://leetcode.com/problems/palindrome-number/
def isPalindrome(x: int) -> bool:
    # calculate reverted number and compare
    if x == 0:
        return True
    if x % 10 == 0:
        return False
    if x < 0:
        return False

    def get_reverted_number(num):
        res = 0
        while num:
            digit = num % 10
            num = int(num / 10)
            res = res * 10 + digit
        return res

    return x == get_reverted_number(x)


# 12. Integer to Roman
# https://leetcode.com/problems/integer-to-roman/
def intToRoman(num: int) -> str:
    # use if-statements
    order = 0
    chars = (('I', 'V'), ('X', 'L'), ('C', 'D'), ('M'))
    res = ''
    while order < 3:
        digit = num % 10
        digit_5 = digit % 5
        digit_rest = int(digit / 5)
        if digit_5 <=3:
            res = digit_5 * chars[order][0] + res
            if digit_rest:
                res = chars[order][1] + res
        elif digit_5 == 4:
            res = chars[order][0] + (chars[order+1][0] if digit_rest else chars[order][1]) + res
        else:
            res = chars[order][1] + res

        num = int(num/10)
        order += 1

    res = chars[3][0] * num  + res
    return res


# 13. Roman to Integer
# https://leetcode.com/problems/roman-to-integer/
def romanToInt(s: str) -> int:
    # take care of the previous roman digit
    rom = {}
    rom['I'] = 1
    rom['V'] = 5
    rom['X'] = 10
    rom['L'] = 50
    rom['C'] = 100
    rom['D'] = 500
    rom['M'] = 1000
    prev, res = 0, 0
    for c in s:
        cur = rom.get(c)
        if cur is None:
            return None
        res += cur
        if cur > prev:
            res -= 2 * prev
        prev = cur

    return res


# 48. Rotate Image
# https://leetcode.com/problems/rotate-image/
def rotate(matrix: List[List[int]]) -> None:
    # iterate through vertical and horizontal directions and calculate new positions for rotated cells
    s = len(matrix) - 1
    hsi = (s + 2) // 2
    hsj = (s + 1) // 2
    for i in range(hsi):
        for j in range(hsj):
            matrix[i][j], matrix[s - j][i],  matrix[s - i][s - j], matrix[j][s - i] = matrix[s - j][i],  matrix[s - i][s - j], matrix[j][s - i], matrix[i][j]


# 118. Pascal's Triangle (not 100)
# https://leetcode.com/problems/pascals-triangle/
def generate(numRows: int) -> List[List[int]]:
    # basic
    res = []
    for i in range(numRows):
        if i == 0:
            res.append([1])
        else:
            inner = []
            for j in range(len(res[i-1]) - 1):
                inner.append(res[i-1][j] + res[i-1][j+1])
            res.append([1] + inner + [1])

    return res


# 169. Majority Element
# https://leetcode.com/problems/majority-element/
def majorityElement(nums: List[int]) -> int:
    # iterating and storing/updating intermediate answer
    if not nums:
        return None
    res = nums[0]
    counter = 1
    for num in nums[1:]:
        if num == res:
            counter +=1
        else:
            counter -=1
            if counter == 0:
                res = num
                counter = 1
    return res


# 202. Happy Number (not 100)
# https://leetcode.com/problems/happy-number/
def isHappy(n: int) -> bool:
    # just basic
    def transform(n):
        res = 0
        while n:
            rem = n % 10
            n = n // 10
            res += rem ** 2
        return res

    history = set()
    while n != 1:
        if n not in history:
            history.add(n)
        else:
            return False
        n = transform(n)

    return True


# 238. Product of Array Except Self
# https://leetcode.com/problems/product-of-array-except-self/
def productExceptSelf(nums: List[int]) -> List[int]:
    # longer solution
    '''
    ind = None
    prod = 1
    for i in range(len(nums)):
        if nums[i] == 0:
            if ind is not None:
                return [0] * len(nums)
            nums[i] = 1
            ind = i
        else:
            prod *= nums[i]

    if ind is not None:
        result = [0] * len(nums)
        result[ind] = prod
        return result
    return [prod // num for num in nums]
    '''
    # iterate from left, and then from right
    left = 1
    results = [1] * len(nums)
    for i in range(len(nums)):
        results[i] = left
        left *= nums[i]
    right = 1
    for i in range(len(nums) - 1, -1, -1):
        results[i] *= right
        right *= nums[i]

    return results


# 240. Search a 2D Matrix II
# https://leetcode.com/problems/search-a-2d-matrix-ii/
def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    # decrement right horizontal and increment top vertical limits until we find the target
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return False
    i_n = len(matrix) - 1
    j_n = 0

    while i_n >= 0 and j_n < len(matrix[0]):
        if target == matrix[i_n][j_n]:
            return True
        if target < matrix[i_n][j_n]:
            i_n -= 1
        else:
            j_n += 1

    return False


# 277. Find the Celebrity (not 100, locked)
# https://leetcode.com/problems/find-the-celebrity/
def findCeleberity(n):
    # find the right candidate by iterating through array once
    # after that, double check the candidate

    def knows(i, j):
        # implemented externally
        pass

    c = 0
    for i in range(n):
        if knows(c, i):
            c = i

    for i in range(n):
        if i == c:
            continue
        if knows(c, i) or not knows(i, c):
            return -1

    return c


# 287. Find the Duplicate Number
# https://leetcode.com/problems/find-the-duplicate-number/
def findDuplicate(nums: List[int]) -> int:
    # first tricky solution
    '''
    l, h = 1, len(nums) - 1
    while l < h:
        m = (l + h) // 2
        count = sum(num <= m for num in nums)
        if count <= m:
            l = m + 1
        else:
            h = m
    return l
    '''
    # detect position of cycle in list

    s = f = nums[0]

    while True:
        s = nums[s]
        f = nums[nums[f]]
        if s == f:
            break
    s = nums[0]
    while s != f:
        s = nums[s]
        f = nums[f]

    return f


# 289. Game of Life (not 100)
# https://leetcode.com/problems/game-of-life/
def gameOfLife(board: List[List[int]]) -> None:
    # basic
    if not board or not board[0]:
        return

    n = len(board)
    m = len(board[0])
    next = [[None for _ in range(m)] for _ in range(n)]

    def calc(i, j):
        live = 0
        dead = 0
        for i1 in [i - 1, i, i + 1]:
            for j1 in [j - 1, j, j + 1]:
                if i1 < 0 or i1 >= n or j1 < 0 or j1 >= m or (i1 == i and j1 == j):
                    continue
                live += board[i1][j1]
                dead += 1 - board[i1][j1]
        if board[i][j]:
            if live < 2 or live > 3:
                return 0
            else:
                return 1
        else:
            return 1 if live == 3 else 0

    for i in range(n):
        for j in range(m):
            next[i][j] = calc(i, j)

    for i in range(n):
        for j in range(m):
            board[i][j] = next[i][j]


# 324. Wiggle Sort II (mot 100)
# https://leetcode.com/problems/wiggle-sort-ii/
def wiggleSort(nums: List[int]) -> None:
    '''
    # find median element (quick select or 2 heaps)
    # place numbers less than medium on odd places, and the rest on even places
    # this solution does not for the case of multiple medians
    from statistics import median
    med = median(nums)
    i = 0
    j = 1
    while True:
        while i < len(nums) and nums[i] <= med:
            i += 2
        while j < len(nums) and nums[j] > med:
            j += 2
        if i >= len(nums) or j >= len(nums):
            break
        nums[i], nums[j] = nums[j], nums[i]
    '''

    # use array of counts

    count = [0] * 5001
    for i in nums:
        count[i] += 1

    c = 5000

    # putting the largest first at the odd site
    i = 0
    while i * 2 + 1 < len(nums):
        while count[c] == 0:
            c -= 1
        nums[i * 2 + 1] = c
        count[c] -= 1
        i += 1

    # putting the largest in the remainder at the even site from largest to smallest.
    i = 0
    while i * 2 < len(nums):
        while count[c] == 0:
            c -= 1
        nums[i * 2] = c
        count[c] -= 1
        i += 1
