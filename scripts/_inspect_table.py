import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
lines = p.read_text(encoding="utf-8").split("\n")
TR = re.compile(r"^\s*\|.*\|\s*$")
SR = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def cells(l):
    s = l.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return re.split(r"(?<!\\)\|", s)


i = 0
while i < len(lines):
    if TR.match(lines[i]) and i + 1 < len(lines) and SR.match(lines[i + 1]):
        h = len(cells(lines[i]))
        j = i + 2
        bad = []
        while j < len(lines) and TR.match(lines[j]):
            n = len(cells(lines[j]))
            if n != h:
                bad.append((j + 1, n, lines[j][:180]))
            j += 1
        if bad:
            print(f"header has {h} cols (line {i+1}):")
            print("   HEADER:", lines[i][:180].encode("ascii", "replace").decode())
            print("   SEP   :", lines[i + 1][:180].encode("ascii", "replace").decode())
            for ln, n, txt in bad[:6]:
                print(f"   line {ln}: {n} cols -> {txt.encode('ascii','replace').decode()}")
            print()
        i = j
    else:
        i += 1
