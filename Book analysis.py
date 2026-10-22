import os, time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------
# 1: SCRAPE THE DATA
# ---------------------------------------------------------------

BASE = "https://books.toscrape.com/catalogue/"

def get_page(url):
    r = requests.get(url)
    r.encoding = "utf-8"
    time.sleep(0.1)
    return BeautifulSoup(r.text, "html.parser")

if os.path.exists("data/books_raw.csv"):
    raw = pd.read_csv("data/books_raw.csv")
else:
    books = []
    for page in range(1, 51):
        soup = get_page(BASE + f"page-{page}.html")
        for item in soup.select("article.product_pod"):
            link = item.h3.a
            book = {
                "title": link["title"],
                "price": item.select_one("p.price_color").text,
                "rating": item.select_one("p.star-rating")["class"][1],
                "stock": item.select_one("p.availability").text.strip(),
            }
            
            detail = get_page(BASE + link["href"])
            book["category"] = detail.select("ul.breadcrumb li a")[2].text
            desc = detail.select_one("#product_description")
            book["description"] = desc.find_next_sibling("p").text if desc else ""
            rows = detail.select("table.table-striped tr")
            info = {r.th.text: r.td.text for r in rows}
            book["upc"] = info["UPC"]
            book["availability"] = info["Availability"]
            books.append(book)
        print("page", page, "done")
    raw = pd.DataFrame(books)
    raw.to_csv("data/books_raw.csv", index=False)

print("Books scraped:", len(raw))

# ---------------------------------------------------------------
# 2: CLEAN THE DATA
# ---------------------------------------------------------------
df = raw.drop_duplicates(subset="upc").copy()                 

df["price"] = df["price"].str.replace(r"[^0-9.]", "", regex=True).astype(float)

stars = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating"] = df["rating"].map(stars)

df["in_stock"] = df["availability"].str.contains("In stock")
df["stock_qty"] = df["availability"].str.extract(r"(\d+)")[0].astype(float).fillna(0)

df["description"] = df["description"].fillna("")
df["has_description"] = (df["description"] != "").astype(int)
df["desc_words"] = df["description"].str.split().str.len()
df["title_words"] = df["title"].str.split().str.len()

df = df[["title", "category", "price", "rating", "in_stock", "stock_qty",
         "has_description", "desc_words", "title_words"]]
df.to_csv("data/books_clean.csv", index=False)
print(df.describe().round(2))

# ---------------------------------------------------------------
# 3: VISUALISE
# ---------------------------------------------------------------

plt.hist(df["price"], bins=30, color="steelblue")
plt.title("Price distribution"); plt.xlabel("Price (£)"); plt.ylabel("Number of books")
plt.savefig("figures/1_price_hist.png"); plt.close()

groups = [df[df["rating"] == r]["price"] for r in [1, 2, 3, 4, 5]]
plt.boxplot(groups)
plt.xticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"])
plt.title("Price by star rating"); plt.xlabel("Stars"); plt.ylabel("Price (£)")
plt.savefig("figures/2_price_by_rating.png"); plt.close()

top10 = df["category"].value_counts().head(10).index
avg = df[df["category"].isin(top10)].groupby("category")["price"].mean().sort_values()
avg.plot(kind="barh", color="steelblue")
plt.title("Average price by genre (10 biggest)"); plt.xlabel("Average price (£)")
plt.tight_layout(); plt.savefig("figures/3_price_by_genre.png"); plt.close()

plt.scatter(df["desc_words"], df["price"], alpha=0.4)
plt.title("Price vs description length"); plt.xlabel("Words in description"); plt.ylabel("Price (£)")
plt.savefig("figures/4_price_vs_description.png"); plt.close()

print("\nCorrelation with price:")
print(df[["price", "rating", "stock_qty", "desc_words", "title_words"]].corr()["price"].round(3))

# ---------------------------------------------------------------
# 4: LINEAR REGRESSION
# ---------------------------------------------------------------

counts = df["category"].value_counts()
df["genre"] = np.where(df["category"].map(counts) >= 15, df["category"], "Other")

# features: numbers + genre dummy columns
X = df[["rating", "stock_qty", "desc_words", "title_words", "has_description"]].astype(float)
X = pd.concat([X, pd.get_dummies(df["genre"], drop_first=True, dtype=float)], axis=1)
y = df["price"].to_numpy()

np.random.seed(42)
order = np.random.permutation(len(df))
cut = int(0.8 * len(df))
train, test = order[:cut], order[cut:]

A = np.column_stack([np.ones(len(X)), X.to_numpy()])
coef, *_ = np.linalg.lstsq(A[train], y[train], rcond=None)

pred = A[test] @ coef            # predictions for the test books
actual = y[test]

mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred) ** 2))
r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)

base = y[train].mean()
base_mae = np.mean(np.abs(actual - base))

print("\n--- Model results (test set) ---")
print(f"R-squared : {r2:.3f}")
print(f"MAE       : £{mae:.2f}   (baseline guess: £{base_mae:.2f})")
print(f"RMSE      : £{rmse:.2f}")

names = ["intercept"] + list(X.columns)
result = pd.Series(coef, index=names).round(3)
print("\nCoefficients (effect on price in £):")
print(result.sort_values(key=abs, ascending=False).head(10))

plt.scatter(actual, pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Predicted vs actual price"); plt.xlabel("Actual (£)"); plt.ylabel("Predicted (£)")
plt.savefig("figures/5_predicted_vs_actual.png"); plt.close()

print("\nDone. Figures are in the 'figures' folder.")
