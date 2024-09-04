import re


regex = re.compile(
    r'(?P<year>(?:\d{4}\-\d{2})|(?:\d{4}))\s+(?P<sport>Basketball|Donruss)?\s*(?P<supplier>Panini|Topps)?\s*(?P<series>Mosaic|Prizm|Optic|Select|Chrome|One\sand\sOne)\s+(?:(Premier\sLeague|UEFA)\s+)?(?P<player>[\w\.\'\s\-]+)\s*(-\s*(?P<team>[\w\d\.\s\-]+))?\s+(?P<no>#\d+)',
    re.IGNORECASE)

tag = '2023-24 Donruss Optic Brook Lopez #15'
regMatch = regex.match(tag)
groupDict = regMatch.groupdict()
print(groupDict)