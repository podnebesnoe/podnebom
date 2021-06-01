from typing import List


# 29. Divide Two Integers
# https://leetcode.com/problems/divide-two-integers/
def divide(dividend: int, divisor: int) -> int:
    # first step: find smallest divisor * 2^n which is greater than dividend (lets call it accumulator)
    # second step: keep deducting accumulator from dividend and reducing accumulator,
    # until dividend is still divisible on divisor

    if divisor == 0:
        return None
    if dividend == 0:
        return 0

    neg = (dividend < 0) ^ (divisor < 0)
    dividend = dividend if dividend > 0 else - dividend
    divisor = divisor if divisor > 0 else - divisor

    pow = 1
    acc = divisor
    while acc < dividend:
        acc += acc
        pow += pow

    res = 0
    while dividend >= divisor:
        if acc <= dividend:
            dividend -= acc
            res += pow
        acc >>= 1
        pow >>= 1

    return max(-res,  -2147483648) if neg else min(res, 2147483647)


# 136. Single Number
# https://leetcode.com/problems/single-number/
# https://www.youtube.com/watch?v=CvnnCZQY2A0
def singleNumber(nums: List[int]) -> int:
    # use xor
    res = 0
    for num in nums:
        res ^= num
    return res


def singleNumberAnother(nums: List[int]) -> int:
    seen_once = 0
    seen_twice = 0
    for num in nums:
        seen_once = ~seen_twice & (seen_once ^ num)
        seen_twice = ~seen_once & (seen_twice ^ num)
    return seen_once


# 231. Power of Two
# https://leetcode.com/problems/power-of-two/
# https://www.youtube.com/watch?v=uVAvuyah9Ek
def isPowerOfTwo(n: int) -> bool:
    # bit solution. can be solved by division on 2
    if not n:
        return False
    while n:
        if n == 1:
            return True
        if n & 1:
            return False
        n >>= 1
    return True


# 338. Counting Bits
# https://leetcode.com/problems/counting-bits/
def countBits(num: int) -> List[int]:
    # repeating pattern of binary representation of increasing numbers with extra 1 in front
    res = [0]

    while len(res) <= num:
        res.extend([i+1 for i in res])

    return res[:num+1]


# 371. Sum of Two Integers (not 100)
# https://leetcode.com/problems/sum-of-two-integers/
def getSum(a: int, b: int) -> int:
    # using ^ for addition without carry, and & << for carry
    while b:
        diff = a^b
        carry = (a&b) << 1
        a = diff
        b = carry
    return a


# 461. Hamming Distance
# https://leetcode.com/problems/hamming-distance/
# https://www.youtube.com/watch?v=oGU1At1GFvc
def hammingDistance(x: int, y: int) -> int:
    # iterate and apply >>. at each iteration, consider last bit
    res = 0
    while x or y:
        res += (x & 1) ^ (y & 1)
        x >>= 1
        y >>= 1
    return res
