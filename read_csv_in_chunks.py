import pandas as pd

chunk_iterable = pd.read_csv("~\\Desktop\\steam_review\\steam_reviews.csv", header="infer", index_col = 0, chunksize= 4*10**6)

count = 0
for chunk in chunk_iterable:
  count += 1

print("\n\n\n\n\n\n", count)
