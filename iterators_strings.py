# 3. Longest Substring Without Repeating Characters
# https://leetcode.com/problems/longest-substring-without-repeating-characters/
def lengthOfLongestSubstring(s: str) -> int:
    # iterate and update answer
    # store value of previous iterators in the dictionary
    inds = {}
    max_len = 0
    ind1 = -1
    for ind2, ch in enumerate(s):
        prev_ind = inds.get(ch)
        if prev_ind is not None and prev_ind > ind1:
            ind1 = prev_ind
        inds[ch] = ind2
        if ind2 - ind1 > max_len:
            max_len = ind2 - ind1
    return max_len

# 5. Longest Palindromic Substring
# https://leetcode.com/problems/longest-palindromic-substring/
def longestPalindrome(s: str) -> str:
    # iterators expanding from the middle
    # treating odd and even cases separately

    if s is None or s =='':
        return ''

    start = 0
    end = 0

    def expand_from_middle(s, left, right):
        while left >=0  and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    for ind in range(len(s)):
        len_cur = expand_from_middle(s, ind, ind)
        if len_cur > end - start:
            start = int(ind - (len_cur - 1)/2)
            end = int(ind + (len_cur - 1)/2 + 1)
        len_cur = expand_from_middle(s, ind, ind+1)
        if len_cur > end - start:
            start = int(ind - (len_cur - 2)/2)
            end = int(ind + (len_cur - 2)/2 + 2)

    return s[start:end]


# 6. ZigZag Conversion
# https://leetcode.com/problems/zigzag-conversion/
def convert(s: str, numRows: int) -> str:
    # organize array into 2D array and then iterate by rows
    if numRows <= 1:
        return s
    rows = [[] for _ in range(numRows)]
    ind = 0
    dir = -1
    for c in s:
        rows[ind].append(c)
        if ind == 0 or ind == numRows - 1:
            dir *= -1
        ind += dir
    return ''.join([''.join(row) for row in rows])


# 71. Simplify Path (not 100)
# https://leetcode.com/problems/simplify-path/
def simplifyPath(path: str) -> str:
    # two iterators (one is for getting next item, another one is for filling)
    # pay attention to split (includes empty strings) and join
    items = [item for item in path.split('/') if item!='' and item!='.']
    i2 = 0
    for i1 in range(len(items)):
        if items[i1] =='..':
            i2 = max(i2 -1, 0)
        else:
            items[i2] = items[i1]
            i2 += 1

    return '/' + '/'.join(items[:i2])


# 647. Palindromic Substrings
# https://leetcode.com/problems/palindromic-substrings/
def countSubstrings(s: str) -> int:
    # iterators expanding from the center
    # consider odd and even cases separately
    ans = len(s)
    # odd cases
    for i in range(len(s)):
        j = 1
        while i - j >= 0 and i + j < len(s):
            if s[i - j] == s[i + j]:
                ans += 1
                j += 1
            else:
                break

    # even cases
    for i in range(len(s) - 1):
        j = 0
        while i - j >= 0 and i + j + 1 < len(s):
            if s[i - j] == s[i + 1 + j]:
                ans += 1
                j += 1
            else:
                break

    return ans
