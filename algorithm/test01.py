from typing import List

def max_len(s: str) -> str:
    str_len = len(s)
    max_len = 0
    index = 0
    for the_len in range(1, str_len + 1):
        for i in range(str_len):
            if i + the_len > str_len:
                break

            if s[i: i + the_len] == s[i: i + the_len][::-1] and the_len > max_len:
                max_len = the_len
                index = i
    return s[index: index + max_len]

print(max_len('babad'))