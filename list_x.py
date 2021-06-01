from typing import List

# auxiliary

class ListNode:
    def __init__(self, val = None):
        self.val = val
        self.next = None

def arr_to_list(nums):
    if len(nums) == 0:
        return None
    root = ListNode(nums[0])
    prev = root
    for num in nums[1:]:
        node = ListNode(num)
        prev.next = node
        prev = node

    return root

def list_to_arr(l):
    nums = []
    while l:
        nums.append(l.val)
        l = l.next
    return nums


# 2. Add Two Numbers
# https://leetcode.com/problems/add-two-numbers/
def addTwoNumbers(l1, l2):
    # iterate through lists and use the carry
    prev = head = None
    carry = 0
    while l1 or l2 or carry:
        val = carry
        if l1:
            val += l1.val
            l1 = l1.next
        if l2:
            val += l2.val
            l2 = l2.next
        carry = int(val/10)
        val = val % 10
        l = ListNode(val)
        if prev:
            prev.next = l
        else:
            head = l
        prev = l
    return head


# 19. Remove Nth Node From End of List
# https://leetcode.com/problems/remove-nth-node-from-end-of-list/
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    # iterate n times by l2 iterator
    # then iterate with l1 and l2 until the end
    # them remove the mode

    l2 = l1 = head
    for i in range(n):
        l2 = l2.next
        if l2 is None:
            if i == n - 1:
                return l1.next
            else:
                return l1
    while l2.next:
        l1 = l1.next
        l2 = l2.next
    l1.next = l1.next.next
    return head


# 21. Merge Two Sorted Lists
# https://leetcode.com/problems/merge-two-sorted-lists/
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    # iterate through both lists and fill merged list
    #head = None
    #prev = None
    dummy = ListNode()
    prev = dummy
    while l1 or l2:
        node = ListNode()
        if l2 is None or (l1 is not None and l1.val <= l2.val):
            node.val = l1.val
            l1 = l1.next
        elif l2 is not None:
            node.val = l2.val
            l2 = l2.next
        #if prev is not None:
        #    prev.next = node
        #else:
        #    head = node
        #prev = node
        prev.next = node
        prev = node
    #return head
    return dummy.next


# 23. Merge k Sorted Lists
# https://leetcode.com/problems/merge-k-sorted-lists/
# https://www.youtube.com/watch?v=zLcNwcR6yO4
def mergeKLists(lists: List[ListNode]) -> ListNode:

    # use previous solution multiple times. takes too much time/
    #if not lists:
    #    return None
    #result = lists[0]
    #for head in lists[1:]:
    #    result = mergeTwoLists(result, head)
    #return result

    # push everything to the heap and pull it out

    import heapq
    heap = []
    for l in lists:
        while l:
            heapq.heappush(heap, l.val)
            l = l.next

    head = l = ListNode()

    while heap:
        val = heapq.heappop(heap)
        l.next = ListNode(val)
        l = l.next

    return head.next

# 24. Swap Nodes in Pairs
# https://leetcode.com/problems/swap-nodes-in-pairs/
def swapPairs(head: ListNode) -> ListNode:
    # use first, second and third iterator
    tmp = ListNode()
    tmp.next = head
    cur = tmp
    while cur.next and cur.next.next:
        first = cur.next
        second = cur.next.next
        third = cur.next.next.next
        cur.next = second
        cur.next.next = first
        cur.next.next.next = third
        cur = cur.next.next
    return tmp.next


# 61. Rotate List (not 100)
# https://leetcode.com/problems/rotate-list/
def rotateRight(head: ListNode, k: int) -> ListNode:
    # calculate length of the list
    # use remainder of k on l as k
    # iterate to new head of the list
    # connect tail to the head, and disconnect element before new head from new head
    if head is None:
        return None
    it = head
    l = 0
    while it:
        it = it.next
        l += 1
    k = k % l
    if k == 0:
        return head
    it = head
    for _ in range(l - k - 1):
        it = it.next

    new_head = it.next
    it.next = None

    it = new_head
    while it.next:
        it = it.next
    it.next = head

    return new_head


# 82. Remove Duplicates from Sorted List II (not 100)
# https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
def deleteDuplicates(head: ListNode) -> ListNode:
    # two iterators: slow and fast which is checking duplicates
    if head is None or head.next is None:
        return head


    dummy = ListNode(head.val-1)
    dummy.next = head

    n1 = dummy
    n2 = n1.next

    while n2:
        if n2.next and n2.val == n2.next.val:
            while n2.next and n2.val == n2.next.val:
                n2 = n2.next
            n1.next = n2.next
        else:
            n1 = n1.next
        n2 = n2.next

    return dummy.next


# 86. Partition List (not 100)
# https://leetcode.com/problems/partition-list/
def partition(head: ListNode, x: int) -> ListNode:
    # form 2 lists from original list using 2 iterators, and then glue them together

    if head is None or head.next is None:
        return head


    head1 = None
    head2 = None

    it1 = None
    it2 = None

    it = head

    while it is not None:

        if it.val < x:
            if head1 is None:
                head1 = it
                it1 = it
            else:
                it1.next = it
                it1 = it
        else:
            if head2 is None:
                head2 = it
                it2 = it
            else:
                it2.next = it
                it2 = it

        it = it.next

    if it2 is not None:
        it2.next = None

    if head1 is not None:
        it1.next = head2
        return head1
    else:
        return head2


# 138. Copy List with Random Pointer
# https://leetcode.com/problems/copy-list-with-random-pointer/
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

def copyRandomList(head: Node) -> Node:
    # use dictionary
    # 1 loop: map old list into the new list
    # 2 loop: use dictionary to assign random for new list
    '''
    cur = head
    head_prev_new = Node(0)
    new = head_prev_new
    d = {}
    while cur:
        node = Node(cur.val)
        new.next = node
        d[cur] = node
        cur = cur.next
        new = new.next

    cur = head
    head_new = head_prev_new.next
    new = head_new
    while cur:
        if cur.random is not None:
            new.random = d[cur.random]
        cur = cur.next
        new = new.next
    return head_new
    '''

    # 3 loops
    # 1 loop: create new nodes and point next of the new list into the next of the old list, and next of the old list into the new list
    # 2 loop: assign random in the new list
    # 3 loop: point next of the new list into the new list

    if head is None:
        return None
    cur = head
    while cur:
        node = Node(cur.val)
        node.next = cur.next
        cur.next = node
        cur = node.next

    cur = head
    while cur:
        new = cur.next
        if cur.random:
            new.random = cur.random.next
        cur = new.next

    cur = head
    while cur:
        new = cur.next
        cur = new.next
        if new.next is not None:
            new.next = new.next.next

    return head.next


# 141. Linked List Cycle
# https://leetcode.com/problems/linked-list-cycle/
def hasCycle(head: ListNode) -> bool:
    # slow and fast iterator, looping until they are the same or some of them is None
    if head is None or head.next is None:
        return False

    slow = fast = head
    slow = slow.next
    fast = fast.next.next
    while slow is not None and fast is not None:
        if slow == fast:
            return True
        slow = slow.next
        fast = fast.next
        if fast is not None:
            fast = fast.next

    return False


# 142. Linked List Cycle II
# https://leetcode.com/problems/linked-list-cycle-ii/
def detectCycle(head: ListNode) -> ListNode:
    # slow and fast iterator, looping until they are the same (lets call it intersect) or some of them is None
    # after that, looping slow iterator from the head and continuing intersect iterator until they are the same
    if head is None or head.next is None:
        return None

    slow = fast = head
    slow = slow.next
    fast = fast.next.next
    intersect = None
    while slow is not None and fast is not None:
        if slow == fast:
            intersect = slow
            break
        slow = slow.next
        fast = fast.next
        if fast is not None:
            fast = fast.next

    if intersect is None:
        return None

    slow = head
    while slow != intersect:
        slow = slow.next
        intersect = intersect.next

    return intersect


# 147. Insertion Sort List (not 100)
# https://leetcode.com/problems/insertion-sort-list/
def insertionSortList(head: ListNode) -> ListNode:
    # use dummy
    # one loop for cur, another loop for placing cur.next into proper position after prev
    if head is None:
        return None

    dummy = ListNode()
    dummy.next = head

    prev = dummy
    cur = head
    while cur:
        if cur.next is not None and cur.next.val < cur.val:
            while prev.next is not None and prev.next.val < cur.next.val:
                prev = prev.next
            tmp = prev.next
            prev.next = cur.next
            cur.next = cur.next.next
            prev.next.next = tmp
            prev = dummy
        else:
            cur = cur.next

    return dummy.next


# 148. Sort List
# https://leetcode.com/problems/sort-list/
def sortList(head: ListNode) -> ListNode:
    # merge sort
    # iterate slow and fast iterator to find half of the list
    # use tmp iterator before slow to split first half of the list
    # call sortList recursively, and after that rearrange 2 sorted halves into merged list using dummy node

    if head is None or head.next is None:
        return head

    fast = slow = tmp = head

    while fast is not None and fast.next is not None:
        tmp = slow
        slow = slow.next
        fast = fast.next.next

    tmp.next = None

    head = sortList(head)
    slow = sortList(slow)

    # merging sorted lists
    cur = dummy = ListNode()

    while head is not None or slow is not None:
        if head is None or (slow is not None and slow.val < head.val):
            cur.next = slow
            slow = slow.next
        else:
            cur.next = head
            head = head.next
        cur = cur.next

    return dummy.next


# 160. Intersection of Two Linked Lists
# https://leetcode.com/problems/intersection-of-two-linked-lists/
# https://www.youtube.com/watch?v=CPXIkMWNn5Q
def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    # iterate pointers from both lists and switch these pointers after reaching the end
    # increment counter each time after switch
    # if pointers match and counter is less than 3, than this is the intersection
    if headA is None or headB is None:
        return None
    ptr1 = headA
    ptr2 = headB
    cnt = 0
    while cnt < 3:
        if ptr1 == ptr2:
            return ptr1
        ptr1 = ptr1.next
        if ptr1 is None:
            cnt += 1
            ptr1 = headB
        ptr2 = ptr2.next
        if ptr2 is None:
            cnt += 1
            ptr2 = headA
    return None


# 203. Remove Linked List Elements (not 100)
# https://leetcode.com/problems/remove-linked-list-elements/
def removeElements(head: ListNode, val: int) -> ListNode:
    # easy, just use one iterator and check it.next
    if head is None:
        return None

    dummy = ListNode(val - 1)
    dummy.next = head

    it = dummy
    while it is not None:
        while it.next is not None and it.next.val == val:
            it.next = it.next.next
        it = it.next

    return dummy.next


# 206. Reverse Linked List
# https://leetcode.com/problems/reverse-linked-list/
# https://www.youtube.com/watch?v=tQur3kprZQk
def reverseList(head: ListNode) -> ListNode:
    # previous and current iterators
    if head is None:
        return None

    prev = None
    cur = head

    while cur is not None:
        next = cur.next
        cur.next = prev
        prev = cur
        cur = next

    return prev


# 234. Palindrome Linked List
# https://leetcode.com/problems/palindrome-linked-list/
def isPalindrome(head: ListNode) -> bool:
    # iterate until the middle of the list using slow and fast: slow is in the middle when fast reaches the end
    # after that, reverse second half of the list
    slow = fast = head

    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

    slow = reverseList(slow)
    fast = head

    while slow is not None:
        if fast.val != slow.val:
            return False
        slow = slow.next
        fast = fast.next
    return True


# 328. Odd Even Linked List (not 100)
# https://leetcode.com/problems/odd-even-linked-list/
def oddEvenList(head: ListNode) -> ListNode:
    # two iterators: even and odd
    # at the end, reconnect even and odd subslists

    if head is None or head.next is None or head.next.next is None:
        return head

    even = head
    odd = head.next

    odd_head = odd

    while even is not None and odd is not None:

        even_tail = even

        even.next = odd.next
        if odd.next is not None:
            odd.next = odd.next.next

        even = even.next
        odd = odd.next

    if even is not None:
        even_tail = even

    even_tail.next = odd_head

    return head


# 445. Add Two Numbers II (not 100)
# https://leetcode.com/problems/add-two-numbers-ii/
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    # dumping lists to stacks
    # then constructing new list from tail to head
    if l1 is None or l2 is None:
        return None
    s1 = []
    while l1:
        s1.append(l1.val)
        l1 = l1.next
    s2 = []
    while l2:
        s2.append(l2.val)
        l2 = l2.next

    next = None
    carry = 0
    while s1 or s2 or carry:
        if s1:
            carry += s1.pop()
        if s2:
            carry += s2.pop()
        cur = ListNode(carry % 10)
        cur.next = next
        next = cur
        carry  = carry // 10

    return next


# 1721. Swapping Nodes in a Linked List (not 100)
# https://leetcode.com/problems/swapping-nodes-in-a-linked-list/
def swapNodes(head: ListNode, k: int) -> ListNode:
    # find these nodes and then swap the values
    if head is None or head.next is None:
        return head
    if k <= 0:
        return head

    first = head
    for _ in range(k-1):
        first = first.next
        if first is None:
            return head

    first_2 = first
    second = head

    while first_2.next is not None:
        first_2 = first_2.next
        second = second.next

    first.val, second.val = second.val, first.val

    return head
