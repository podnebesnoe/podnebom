from typing import List


class TreeNode:
    def __init__(self, val = None):
        self.val = val
        self.left = None
        self.right = None

def createBTree(data, index):
    pNode = None
    if index < len(data):
        if data[index] == None:
            return
        pNode = TreeNode(data[index])
        pNode.left = createBTree(data, 2 * index + 1) # [1, 3, 7, 15, ...]
        pNode.right = createBTree(data, 2 * index + 2) # [2, 5, 12, 25, ...]
    return pNode


# 94. Binary Tree Inorder Traversal
# https://leetcode.com/problems/binary-tree-inorder-traversal/
def inorderTraversal(root: TreeNode) -> List[int]:
    '''
    # recursively
    def traversal(node, result):
        if node is not None:
            traversal(node.left, result)
            result.append(node.val)
            traversal(node.right, result)

    result = []
    traversal(root, result)
    return result
    '''

    # using stack
    # append to the stack all the left children of cur
    # pop from the stack and then assign cur to right element

    if root is None:
        return []

    result = []
    stack = []
    cur = root
    while cur is not None or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        result.append(cur.val)
        cur = cur.right

    return result


# 96. Unique Binary Search Trees
# https://leetcode.com/problems/unique-binary-search-trees/
def numTrees(n: int) -> int:
    # recursion
    '''
    if n == 0:
        return 1
    num = 0
    for i in range(n):
        num += numTrees(i) * numTrees(n - i - 1)
    return num
    '''
    # dynamic programming
    # we observe that we can reuse solution for smaller left and right subtrees
    nums = [1] + [0] * n
    for i in range(1, n + 1):
        num = 0
        for j in range(i):
            num += nums[j] * nums[i - j - 1]
        nums[i] = num
    return nums[-1]


# 98. Validate Binary Search Tree
# https://leetcode.com/problems/validate-binary-search-tree/
def isValidBST(root: TreeNode) -> bool:
    # recursive caller of a helper function
    def isValid(node, left, right):
        if node is None:
            return True
        if left is not None:
            if node.val <= left:
                return False
        if right is not None:
            if node.val >= right:
                return False
        return isValid(node.left, left, node.val) and isValid(node.right, node.val, right)

    return isValid(root, None, None)


# 99. Recover Binary Search Tree (not 100)
# https://leetcode.com/problems/recover-binary-search-tree/
def recoverTree(root: TreeNode) -> None:
    # use the fact, that in-order traversal for bst should result in sorted array
    # use the fact that if 2 elements are swapped in sorted array,
    # then they can be found by iterating through array and collecting first and second entry where cur <= prev

    if root is None or (root.left is None and root.right is None):
        return

    class Self:
        def __init__(self):
            self.prev = None
            self.first = None
            self.second = None

    self = Self()

    def inorder(node):
        if node is None:
            return False

        inorder(node.left)

        if self.prev is not None:
            if self.first is None:
                if node.val <= self.prev.val:
                    self.first = self.prev
            if node.val <= self.prev.val:
                self.second = node

        self.prev = node

        inorder(node.right)

    inorder(root)

    self.first.val, self.second.val = self.second.val, self.first.val


# 101. Symmetric Tree
# https://leetcode.com/problems/symmetric-tree/
# https://www.youtube.com/watch?v=K7LyJTWr2yA
def isSymmetric(root: TreeNode) -> bool:
    # recursive call of a helper function
    if root is None:
        return True

    def ifBranchesEqual(left, right):
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False
        if left.val != right.val:
            return False
        return ifBranchesEqual(left.left, right.right) and ifBranchesEqual(left.right, right.left)

    return ifBranchesEqual(root.left, root.right)


# 102. Binary Tree Level Order Traversal
# https://leetcode.com/problems/binary-tree-level-order-traversal/
def levelOrder(root: TreeNode) -> List[List[int]]:
    # level order traversal, remember level and increment it for children
    if not root:
        return []
    res = []
    nodes = [(root, 0)]
    while len(nodes) > 0:
        node, level = nodes.pop(0)
        if len(res) <= level:
            res.append([])
        res[level].append(node.val)
        if node.left:
            nodes.append((node.left, level + 1))
        if node.right:
            nodes.append((node.right, level + 1))
    return res


# 103. Binary Tree Zigzag Level Order Traversal (not 100)
# https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
def zigzagLevelOrder(root: TreeNode) -> List[List[int]]:
    # level-order traversal. keep layers, and then reverse layers of the odd level
    if root is None:
        return []

    result = []
    stack = [(root, 0)]
    while stack:
        node, level = stack.pop(0)

        if level >= len(result):
            result.append([])
        result[level].append(node.val)

        if node.left is not None:
            stack.append((node.left, level+1))
        if node.right is not None:
            stack.append((node.right, level+1))

    for i in range(1, len(result), 2):
        result[i] = list(reversed(result[i]))

    return result


# 104. Maximum Depth of Binary Tree
# https://leetcode.com/problems/maximum-depth-of-binary-tree/
def maxDepth(root: TreeNode) -> int:
    # recursive helper function:
    if root is None:
        return 0
    left = maxDepth(root.left) + 1
    right = maxDepth(root.right) + 1
    return max(left, right)


# 105. Construct Binary Tree from Preorder and Inorder Traversal
# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
def buildTree(preorder: List[int], inorder: List[int]) -> TreeNode:
    '''
    # the same solution as below but with list copying, leetcode complained about exceeding time
    if len(preorder) < 1:
        return None
    root = TreeNode()
    root.val = preorder[0]

    ind = inorder.index(preorder[0])
    left_inorder = inorder[:ind]

    ind_right = 1
    while ind_right < len(preorder):
        if preorder[ind_right] not in left_inorder:
            break
        ind_right+=1

    #left_preorder = list(filter(lambda x: x in left_inorder, preorder))
    left_preorder = preorder[1:ind_right]
    root.left = self.buildTree(left_preorder, left_inorder)

    right_inorder = inorder[ind+1:]
    #right_preorder = list(filter(lambda x: x in right_inorder, preorder))
    right_preorder = preorder[ind_right:]
    root.right = self.buildTree(right_preorder, right_inorder)

    return root
    '''

    # observe where indices for left and right subtrees aer located
    # recursive call of helper function with extra indices
    def builTreeWithInd(start_preorder, start_inorder, length, preorder, inorder):
        if length < 1:
            return None
        root = TreeNode()
        root.val = preorder[start_preorder]

        length_left = inorder[start_inorder:start_inorder+length].index(root.val)
        length_right = length - length_left - 1

        root.left = builTreeWithInd(start_preorder + 1, start_inorder, length_left, preorder, inorder)
        root.right = builTreeWithInd(start_preorder + length_left + 1, start_inorder + length_left + 1, length_right, preorder, inorder)

        return root

    return builTreeWithInd(0, 0, len(preorder), preorder, inorder)


# 108. Convert Sorted Array to Binary Search Tree
# https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
# https://www.youtube.com/watch?v=PZYTs9y4f4o
def sortedArrayToBST(nums: List[int]) -> TreeNode:
    # recursive call of itself. notice that root of BST will be at the middle of sorted array
    if not nums:
        return None

    ind = len(nums) // 2
    node = TreeNode(nums[ind])
    node.left = sortedArrayToBST(nums[:ind])
    node.right = sortedArrayToBST(nums[ind+1:])
    return node


# 112. Path Sum
# https://leetcode.com/problems/path-sum/
# https://www.youtube.com/watch?v=Hg82DzMemMI
def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    # recursive call of itself
    if root is None:
        return False
    if root.left is None and root.right is None:
        if root.val == targetSum:
            return True
    return hasPathSum(root.left, targetSum - root.val) or hasPathSum(root.right, targetSum - root.val)


# 113. Path Sum II (not 100)
# https://leetcode.com/problems/path-sum-ii/
def pathSum(root: TreeNode, targetSum: int) -> List[List[int]]:
    # recursive call of helper function and external list for collecting results
    if root is None:
        return None
    res = []
    path = []

    def fun(node, cur):
        if node.left is not None:
            path.append(node.val)
            fun(node.left, cur - node.val)
            path.pop()
        if node.right is not None:
            path.append(node.val)
            fun(node.right, cur - node.val)
            path.pop()

        if node.left is None and node.right is None and cur == node.val:
            res.append(path + [node.val])

    fun(root, targetSum)

    return res


# 114. Flatten Binary Tree to Linked List
# https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
def flatten(root: TreeNode) -> None:
    '''
    # solution without stack
    def helper(node):
        node_right = node.right
        node_left_right = None
        if node.left is not None:
            end = helper(node.left)
            end.right = node_right
            node.right = node.left
            node.left = None
            node_left_right = end
        if node_right is not None:
            return helper(node_right)
        elif node_left_right is not None:
            return node_left_right
        return node

    if root is not None:
        helper(root)
    '''

    # use stack and make the latest element of the stack as a right subtree of the current node

    if root is None:
        return

    stack = [root]
    while len(stack) > 0:
        cur = stack.pop()
        if cur.right is not None:
            stack.append(cur.right)
        if cur.left is not None:
            stack.append(cur.left)

        if len(stack) > 0:
            cur.right = stack[-1]

        cur.left = None


class NextNode:
    def __init__(self, val: int = 0, left: 'NextNode' = None, right: 'NextNode' = None, next: 'NextNode' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


# 116. Populating Next Right Pointers in Each Node (not 100)
# https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
def connect(root: 'NextNode') -> 'NextNode':
    # use level order traversal
    if root is None:
        return root
    root.next = None
    stack = [[root, 0]]
    while stack:
        node, level = stack.pop(0)
        if stack:
            if stack[0][1] == level:
                stack[0][0].next = node
            else:
                stack[0][0].next = None
        if node.left is not None:
            stack.append([node.right, level+1])
            stack.append([node.left, level + 1])
    return root


# 129. Sum Root to Leaf Numbers (not 100)
# https://leetcode.com/problems/sum-root-to-leaf-numbers/
def sumNumbers(root: TreeNode) -> int:
    # recursive call with updating result stored externally
    res = [0]

    def helper(node, num):
        num = num + node.val
        if node.left is None and node.right is None:
            res[0] += num
        else:
            if node.left is not None:
                helper(node.left, num * 10)
            if node.right is not None:
                helper(node.right, num * 10)

    helper(root, 0)

    return res[0]


# 144. Binary Tree Preorder Traversal (not 100)
# https://leetcode.com/problems/binary-tree-preorder-traversal/
def preorderTraversal(root: TreeNode) -> List[int]:
    # just preorder traversal. recursvie function updating the result
    res = []
    def fun(node):
        if node is not None:
            res.append(node.val)
            fun(node.left)
            fun(node.right)
    fun(root)
    return res


# 145. Binary Tree Postorder Traversal
# https://leetcode.com/problems/binary-tree-postorder-traversal/
# https://www.youtube.com/watch?v=sMI4RBEZyZ4
def postorderTraversal(root: TreeNode) -> List[int]:
    # popping out of the end of the stack, not from the beginning (like in the case of level order traversal)
    # inserting at the beginning of resulting list, since this is postorder

    if root is None:
        return None

    result = []

    stack = [root]
    while stack:
        node = stack.pop()
        result.insert(0, node.val)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    return result


# 199. Binary Tree Right Side View
# https://leetcode.com/problems/binary-tree-right-side-view/
# https://www.youtube.com/watch?v=jCqIr_tBLKs
def rightSideView(root: TreeNode) -> List[int]:
    '''
    # some type of traversal: include level information into the stack
    # this helps to find out the right side view (where level popping our from the stack is more than current level)

    if root is None:
        return []

    res = []
    stack = [(root, 0)]
    curlev = -1
    while stack:
        node, level = stack.pop()
        if level > curlev:
            curlev = level
            res.append(node.val)
        if node.left is not None:
            stack.append[(node.left, level + 1)]
        if node.right is not None:
            stack.append[(node.right, level + 1)]

    return res
    '''
    # level-order traversal with extra for
    result = []
    if root is None:
        return result
    queue = [root]
    while queue:
        size = len(queue)
        for ind in range(size):
            node = queue.pop(0)
            if ind == size-1:
                result.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

    return result


# 222. Count Complete Tree Nodes (not 100)
# https://leetcode.com/problems/count-complete-tree-nodes/
def countNodes(root: TreeNode) -> int:
    # recursive call of itself, which has recursive call of helper function for left and right
    if root is None:
        return 0
    def countLeft(node):
        if node is None:
            return 0
        return 1 + countLeft(node.left)
    def countRight(node):
        if node is None:
            return 0
        return 1 + countRight(node.right)
    l = countLeft(root)
    r = countRight(root)
    if l == r:
        return 2**l - 1
    return 1 + countNodes(root.left) + countNodes(root.right)


# 226 Invert Binary Tree
# https://leetcode.com/problems/invert-binary-tree/
def invertTree(root: TreeNode) -> TreeNode:
    # recursive call of itself
    if root is not None:
        root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root


# 230. Kth Smallest Element in a BST
# https://leetcode.com/problems/kth-smallest-element-in-a-bst/
# https://www.youtube.com/watch?v=C6r1fDKAW_o
def kthSmallest(root: TreeNode, k: int) -> int:
    # recursive call of a helper function for in-order traversal
    # keeping order of an element in a result, as well as its value (for final update)

    if root is None or k == 0:
        return root
    result = [k, None]

    def in_order(node, result):
        if node is None:
            return False
        if in_order(node.left, result):
            return True
        result[0] -= 1
        if result[0] == 0:
            result[1] = node.val
            return True
        return in_order(node.right, result)

    in_order(root, result)

    return result[1]


# 235. Lowest Common Ancestor of a Binary Search Tree
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
# https://www.youtube.com/watch?v=kulWKd3BUcI
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # chose right subtree to look and recursive call itself
    if p.val < root.val and q.val < root.val:
            return lowestCommonAncestor(root.left, p, q)
    elif p.val > root.val and q.val > root.val:
        return lowestCommonAncestor(root.right, p, q)
    else:
        return root


# 236. Lowest Common Ancestor of a Binary Tree
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # recursive call itself and correct conditions on pointers for left and right being none or not none
    if root is None:
        return None
    if root.val == p.val or root.val == q.val:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left is not None and right is not None:
        return root
    if left is None and right is None:
        return None
    return left if left is not None else right


# 257. Binary Tree Paths
# https://leetcode.com/problems/binary-tree-paths/
# https://www.youtube.com/watch?v=xqS8dyexaNM
def binaryTreePaths(root: TreeNode) -> List[str]:
    # solution with multiple arrays returned recursively
    '''
    if root.left is None and root.right is None:
        return [str(root.val)]
    result = []
    if root.left is not None:
        result.extend([f'{root.val}->{string}' for string in binaryTreePaths(root.left)])
    if root.right is not None:
        result.extend([f'{root.val}->{string}' for string in binaryTreePaths(root.right)])
    return result
    '''

    # recursive call of a helper function
    # adding path to array if this is the end leaf

    if root is None:
        return []
    paths = []
    def dfs(node, path):
        path += str(node.val)
        if node.left is None and node.right is None:
            paths.append(path)
        if node.left is not None:
            dfs(node.left, f'{path}->')
        if node.right is not None:
            dfs(node.right, f'{path}->')

    dfs(root, "")
    return paths


# 337. House Robber III
# https://leetcode.com/problems/house-robber-iii/
def rob(root: TreeNode) -> int:
    # we can use dynamic programming, but we use dfs instead with returning tuple of 2 possibilities
    # and the return we return maximum of tuple returned for root

    def dfs(node):
        if node is None:
            return (0, 0)

        left_with_root, left_without_root = dfs(node.left)
        right_with_root, right_without_root = dfs(node.right)

        with_root = node.val + left_without_root + right_without_root
        without_root = max(left_with_root, left_without_root) + max(right_with_root, right_without_root)

        return with_root, without_root

    with_root, without_root = dfs(root)
    return max(with_root, without_root)


# 404. Sum of Left Leaves
# https://leetcode.com/problems/sum-of-left-leaves/
# https://www.youtube.com/watch?v=_gnyuO2uquA
def sumOfLeftLeaves(root: TreeNode) -> int:
    # recursive call of a helper function which update the result
    res = [0]

    def helper(node, is_left):
        if node is not None:
            if node.left is None and node.right is None and is_left:
                res[0] += node.val
            helper(node.left, True)
            helper(node.right, False)

    helper(root, False)

    return res[0]


# 437. Path Sum III
# https://leetcode.com/problems/path-sum-iii/
def pathSum(root: TreeNode, sum: int) -> int:
    # recursive call of inner function from root, and of outer function from left and right

    if root is None:
        return 0

    def pathSumRoot(node, tmp):
        if node is None:
            return 0

        res = 0
        if node.val == tmp:
            res += 1
        res += pathSumRoot(node.left, tmp - node.val) + pathSumRoot(node.right, tmp - node.val)
        return res

    return pathSumRoot(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum)

    '''
    # recursively returning all paths from each child and updating answer
    result = [0]
    def helper(node):
        if node is None:
            return []

        arr = []
        if node.left is not None:
            arr.extend(helper(node.left))
        if node.right is not None:
            arr.extend(helper(node.right))

        arr = [item + node.val for item in arr]
        arr += [node.val]
        for item in arr:
            if item == sum:
                result[0] += 1
        return arr

    helper(root)

    return result[0]
    '''


# 450. Delete Node in a BST (not 100)
# https://leetcode.com/problems/delete-node-in-a-bst/
def deleteNode(root: TreeNode, key: int) -> TreeNode:
    # recursive function.
    # bst: using ultimate right of the left node to put its value for the root
    if root is None:
        return None
    if root.val == key:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        tmp = root.left
        while tmp.right is not None:
            tmp = tmp.right
        root.val = tmp.val
        root.left = deleteNode(root.left, root.val)
    elif root.val > key:
        root.left = deleteNode(root.left, key)
    else:
        root.right = deleteNode(root.right, key)
    return root


# 508. Most Frequent Subtree Sum (not 100)
# https://leetcode.com/problems/most-frequent-subtree-sum/
def findFrequentTreeSum(root: TreeNode) -> List[int]:
    # calculate subtree sums recursively and build a dictionary for them (key - subtree sum, value - number of entrances)
    # using this dictionary, find subtree sums with most number of entrances
    if root is None:
        return []

    d = {}

    def helper(node):
        val = node.val
        if node.left:
            val += helper(node.left)
        if node.right:
            val += helper(node.right)

        if val not in d:
            d[val] = 0
        d[val] += 1

        return val

    helper(root)

    vals = sorted(d.items(), key = lambda x: x[1], reverse = True)

    num = vals[0][1]

    ind = 0
    while ind < len(vals) and vals[ind][1] == num:
        ind += 1

    return [x[0] for x in vals[:ind]]


# 538. Convert BST to Greater Tre (not 100)
# https://leetcode.com/problems/convert-bst-to-greater-tree/
def convertBST(self, root: TreeNode) -> TreeNode:
    # traverse tree using modified in order, so it will be iterating from largest to smallest values
    # keep updating sum and assign it for the each node
    s = [0]
    def helper(node):
        if node is None:
            return
        helper(node.right)
        s[0] += node.val
        node.val = s[0]
        helper(node.left)

    helper(root)
    return root


# 543. Diameter of Binary Tree
# https://leetcode.com/problems/diameter-of-binary-tree/
def diameterOfBinaryTree(root: TreeNode) -> int:
    # calculating longest path and at the same time updating maximal diameter
    res = [0]

    def getMaxLength(node):
        if node is None:
            return 0
        left = getMaxLength(node.left)
        right = getMaxLength(node.right)
        res[0] = max(res[0], left + right)
        return max(left, right) + 1

    getMaxLength(root)

    return res[0]


# 617. Merge Two Binary Trees
# https://leetcode.com/problems/merge-two-binary-trees/
def mergeTrees(root1: TreeNode, root2: TreeNode) -> TreeNode:
    # recursive call for both trees
    if root1 is None and root2 is None:
        return None

    root = TreeNode(0)
    if root1 is not None:
        root.val += root1.val
    if root2 is not None:
        root.val += root2.val

    root.left = mergeTrees(root1.left if root1 else None, root2.left if root2 else None)
    root.right = mergeTrees(root1.right if root1 else None, root2.right if root2 else None)

    return root


# 938. Range Sum of BST
# https://leetcode.com/problems/range-sum-of-bst/
# https://www.youtube.com/watch?v=SIdrJwWp3H0
def rangeSumBST(root: TreeNode, low: int, high: int) -> int:
    # recursive call of a helper function, which does not allow to go to the branches outside of the [low, high]

    sum = [0]

    def recursive(node, left_limit, right_limit):
        if node is None:
            return
        if (left_limit is not None and left_limit > high) or (right_limit is not None and right_limit < low):
            return
        if node.val >= low and node.val <= high:
            sum[0] += node.val
        recursive(node.left, left_limit, node.val)
        recursive(node.right, node.val, right_limit)

    recursive(root, low, high)

    return sum[0]


# 1110. Delete Nodes And Return Forest
# https://leetcode.com/problems/delete-nodes-and-return-forest/
# https://www.youtube.com/watch?v=aaSFzFfOQ0o
def delNodes(root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
    # recursive call to helper function which returns None or node
    # filling the result at each recursive call

    to_delete = set(to_delete)

    remaining = []

    def remove_nodes(node, to_delete, remaining):
        if node is None:
            return None
        node.left = remove_nodes(node.left, to_delete, remaining)
        node.right = remove_nodes(node.right, to_delete, remaining)
        if node.val in to_delete:
            if node.left is not None:
                remaining.append(node.left)
            if node.right is not None:
                remaining.append(node.right)
            return None
        else:
            return node

    node = remove_nodes(root, to_delete, remaining)
    if node is not None:
        remaining.append(node)

    return remaining


# 1740 (locked)
# https://leetcode.com/problems/find-distance-in-a-binary-tree/
def distance(root, data1, data2):
    # find lowest common ancestor (see above)
    # after that, just calculate distances for the nodes
    lca = lowestCommonAncestor(root, data1, data2)

    def distance(root, data):
        if root is None:
            return -1
        if root.val == data.val:
            return 0
        left = distance(root.left, data)
        if left >= 0:
            return left + 1
        right = distance(root.right, data)
        if right >= 0:
            return right + 1
        return -1

    return distance(root, data1) + distance(root, data2) - 2 * distance(root, lca)
