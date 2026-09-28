from ddgs import DDGS

results = DDGS().text("Amazon ML Challenge business entity resolution", max_results=5)

for r in results:
    print(r["title"])
    print(r["href"])
    print(r["body"][:120])
    print()