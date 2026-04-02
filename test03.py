from typing import List

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        ans = [1]*len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    ans[i] = max(ans[i], ans[j]+1)
        print(ans)
        return max(ans)

data = [10,9,2,5,3,7,101,18]
print(Solution().lengthOfLIS(data))
