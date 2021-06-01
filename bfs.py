from typing import List


# 403. Frog Jump
# https://leetcode.com/problems/frog-jump/
# https://www.youtube.com/watch?v=4LvYp_d6Ydg&t=1s
def canCross(stones: List[int]) -> bool:
    # use temporary lists with current positions and jumps
    # loop: append for next position/jump and pop current position/jump under consideration

    for i in range(2, len(stones)):
        if stones[i] > 2 * stones[-1]:
            return False

    last_position = stones[-1]
    stones = set(stones)
    positions = [0]
    jumps = [0]

    while positions:
        position = positions.pop()
        jump = jumps.pop()
        for next_jump in (jump-1, jump, jump+1):
            if next_jump <= 0:
                continue
            next_position = position + next_jump
            if next_position == last_position:
                return True
            elif next_position in stones:
                positions.append(next_position)
                jumps.append(next_jump)
    return False
