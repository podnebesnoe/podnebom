from typing import List


# 14. Longest Common Prefix
# https://leetcode.com/problems/longest-common-prefix/
def longestCommonPrefix(strs: List[str]) -> str:
    # iterate through all strings
    if len(strs) < 1:
        return ''
    ind = 0
    while True:
        c = None
        for string in strs:
            if ind >= len(string):
                return string[:ind]
            if c is None:
                c = string[ind]
            elif c != string[ind]:
                return string[:ind]
        ind += 1


# 43. Multiply Strings
# https://leetcode.com/problems/multiply-strings/
# similar problem but for adding:
# https://www.youtube.com/watch?v=_Qp-CTzat50
def multiply(num1: str, num2: str) -> str:
    # multiply numbers like at school and collect the answer in the list
    # ''.join(list)
    if not len(num1) or not len(num2):
        return ''
    res = [0] * (len(num1) + len(num2))
    for i in range(len(num1) - 1, -1, -1):
        for j in range(len(num2) - 1, -1, -1):
            carry = (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
            ind = i + j + 1
            while carry:
                res[ind] += carry
                if res[ind] >= 10:
                    carry = res[ind] // 10
                    res[ind] = res[ind] % 10
                    ind -=1
                else:
                    carry = 0

    while ind < len(res) and res[ind] == 0:
        ind += 1
    return ''.join([str(i) for i in res[ind:]]) if ind < len(res) else '0'


# 49. Group Anagrams
# https://leetcode.com/problems/group-anagrams/
# https://www.youtube.com/watch?v=ptgykfAEax8
def groupAnagrams(strs: List[str]) -> List[List[str]]:
    # use dictionary for sorted strings
    d = {}
    for string in strs:
        key = str(sorted(string))
        if key not in d:
            d[key] = []
        d[key].append(string)
    result = []
    for key, val in d.items():
        result.append(val)
    return result


# 387. First Unique Character in a String
# https://leetcode.com/problems/first-unique-character-in-a-string/
# https://www.youtube.com/watch?v=St47WCbQa9M
def firstUniqChar(s: str) -> int:
    # use dictionary for first entrance (or length of the string if more entrances)
    d = {}
    for ind, c in enumerate(s):
        if c not in d:
            d[c] = ind
        else:
            d[c] = len(s)
    arr = list((d.values()))
    res = min(arr)
    return res if res < len(s) else -1


# 394. Decode String
# https://leetcode.com/problems/decode-string/
def decodeString(s: str) -> str:
    # remembering start, end indices for numbers and brackets, and calling itself recursively
    result = ''
    level = 0
    ind_numeric = None
    ind_level = None
    number = 0
    for i in range(len(s)):
        if level == 0:
            if s[i].isnumeric():
                if ind_numeric is None:
                    ind_numeric = i
            elif s[i] == '[':
                number = int(s[ind_numeric:i])
                ind_numeric = None
                ind_level = i + 1
                level += 1
            else:
                result += s[i]
        else:
            if s[i] == '[':
                level += 1
            elif s[i] == ']':
                level -= 1
                if level == 0:
                    result += number * decodeString(s[ind_level:i])
                    ind_level = None
    return result


# 438. Find All Anagrams in a String
# https://leetcode.com/problems/find-all-anagrams-in-a-string/
def findAnagrams(s: str, p: str) -> List[int]:
    # sliding window of the same size
    # deduct beginning of the window and add end of the window

    if len(s) < len(p):
        return []

    arr = [0] * 28
    for c in p:
        arr[ord(c) - ord('a')] += 1

    for i in range(len(p)):
        arr[ord(s[i]) - ord('a')] -= 1

    res = []
    if not any(arr):
        res.append(0)

    for i in range(1, len(s) - len(p) + 1):
        arr[ord(s[i - 1]) - ord('a')] += 1
        arr[ord(s[i + len(p) - 1]) - ord('a')] -= 1
        print(s[i:i + len(p)])
        if not any(arr):
            res.append(i)

    return res


# 451. Sort Characters By Frequency
# https://leetcode.com/problems/sort-characters-by-frequency/
# https://www.youtube.com/watch?v=trFw8IDw2Vg&t=3s
def frequencySort(s: str) -> str:
    # use dictionary of entrances
    d = {}
    for c in s:
        if c not in d:
            d[c] = 0
        d[c] += 1

    arr = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return ''.join([k * v for v, k in arr])


# 844. Backspace String Compare
# https://leetcode.com/problems/backspace-string-compare/
# https://www.youtube.com/watch?v=96-d8ZPjHeE
def backspaceCompare(s: str, t: str) -> bool:
    # very basic
    def transform(string):
        res = ''
        for ch in string:
            if ch == '#':
                res = res[:-1]
            else:
                res += ch
        return res

    return transform(s) == transform(t)
