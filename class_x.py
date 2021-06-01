class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


# 146. LRU Cache
# https://leetcode.com/problems/lru-cache/
class LRUCache:
    # double-linked list and dictionary (pointing on its elements)
    # tail is actually dummy node before head

    class Node:
        def __init__(self, val, prev=None, next=None):
            self.val = val
            self.key = None
            self.prev = prev
            self.next = next

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.d = {}
        self.head = self.Node(None, None)
        self.tail = self.Node(None, None)
        #self.tail.next = self.head
        self.head.prev = self.tail

    def remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def add(self, val):
        node = self.Node(val)
        node.next, node.prev = self.head, self.head.prev
        self.head.prev.next = node
        self.head.prev = node
        return node

    def get(self, key: int) -> int:
        node = self.d.get(key)
        if node is None:
            return -1
        self.remove(node)
        self.d[key] = self.add(node.val)
        self.d[key].key = key
        return node.val


    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return
        node = self.d.get(key)
        if node is not None:
            self.remove(node)
        elif len(self.d) >= self.capacity:
            del self.d[self.tail.next.key]
            self.remove(self.tail.next)
        self.d[key] = self.add(value)
        self.d[key].key = key


# 155. Min Stack
# https://leetcode.com/problems/min-stack/
class MinStack:
    # one stack for elements, another stack just for mins

    def __init__(self):
        self.stack = []
        self.mins = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.mins or val <= self.mins[-1]:
            self.mins.append(val)

    def pop(self) -> None:
        if not self.stack:
            return None
        val = self.stack.pop()
        if self.mins and val == self.mins[-1]:
            self.mins.pop()
        return val

    def top(self) -> int:
        return self.stack[-1] if self.stack else None

    def getMin(self) -> int:
        return self.mins[-1] if self.mins else None


# 173. Binary Search Tree Iterator (not 100)
# https://leetcode.com/problems/binary-search-tree-iterator/
class BSTIterator:
    # using stack for parent nodes (in order to go up)
    # filling the stack: go right and then go ultimate left
    # use dummy parent node for the root

    def __init__(self, root: TreeNode):
        dummy = TreeNode()
        dummy.right = root
        self.stack = [dummy]
        self.next()

    def next(self) -> int:
        node = self.stack.pop()
        res = node.val
        if node.right is not None:
            node = node.right
            self.stack.append(node)
            while node.left is not None:
                node = node.left
                self.stack.append(node)
        return res

    def hasNext(self) -> bool:
        return self.stack


# 208. Implement Trie (Prefix Tree)
# https://leetcode.com/problems/implement-trie-prefix-tree/
class Trie:
    # tree, where each node is a tuple:
    # first element - if this is the word, second element - array for 28 children

    def __init__(self):
        self.root = [True, [None] * 28]

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if node[1][ord(ch) - ord('a')] is None:
                node[1][ord(ch) - ord('a')] = [False, [None] * 28]
            node = node[1][ord(ch) - ord('a')]
        node[0] = True


    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if node[1][ord(ch) - ord('a')] is None:
                return False
            node = node[1][ord(ch) - ord('a')]
        return node[0]

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if node[1][ord(ch) - ord('a')] is None:
                return False
            node = node[1][ord(ch) - ord('a')]
        return True


# 346 (locked)
# https://leetcode.com/problems/moving-average-from-data-stream/
# https://www.youtube.com/watch?v=E-kjYOZEBxY
class MovingAverage:
    # store all elements of the moving window in order to deduct earliest element from the moving sum

    def __init__(self, size):
        self.sum = 0
        self.nums = []
        self.size = size

    def next(self, val):
        self.nums.append(val)
        self.sum +=val
        if len(self.nums)>self.size:
            self.nums.pop(0)
            self.sum -= self.nums[0]
        return self.sum/self.size

