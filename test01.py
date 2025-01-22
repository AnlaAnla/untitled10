from Levenshtein import distance


def normalized_levenshtein_similarity(str1, str2):
    """
    计算两个字符串的归一化 Levenshtein 相似度，结果在 0-1 之间。
    """
    dist = distance(str1, str2)
    max_len = max(len(str1), len(str2))

    if max_len == 0:
        return 1  # 两个空字符串相似度为 1
    else:
        return 1 - (dist / max_len)


name_dis_list = [
    ['2023-24 PANINI CROWN ROYALE BASE', 0.5731005495334103],
    ['2023-24 PANINI CROWN ROYALE BASE', 0.5488925433467327],
    ['2023-24 PANINI COURT KINGS ROOKIES III', 0.541679716087909],
    ['2023-24 PANINI DONRUSS DONRUSS-BASE', 0.5330649817136432],
    ['2023-24 PANINI CROWN ROYALE MONARCH MEMORABILIA', 0.5328497052023063],
    ['2023-24 PANINI SELECT SELECT CERTIFIED', 0.5093153238525343],
    ['2023-24 PANINI COURT KINGS STATE OF THE ART', 0.5010120226459216],
    ['2023-24 PANINI DONRUSS ELITE ELITE SERIES BLUE', 0.5003761460413763],
    ['2023-24 PANINI SELECT EN FUEGO', 0.49395430870571444],
    ['2023-24 PANINI SELECT EN FUEGO', 0.4928340253249842]
]

for i in range(len(name_dis_list)):
    dist = normalized_levenshtein_similarity(name_dis_list[i][0],
                                             'CROWN ROAYEL')
    print(dist)
    name_dis_list[i][1] += dist

sorted_name_dis_list = sorted(name_dis_list, key=lambda item: item[1], reverse=True)

for item in sorted_name_dis_list:
    print(item)
