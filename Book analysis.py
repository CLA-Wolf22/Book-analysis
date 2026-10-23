
import os
import time
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------
# 1: SCRAPE THE DATA
# ---------------------------------------------------------------

BASE = "https://books.toscrape.com/catalogue/page-1.html"
RAW_FILE = "data/books_raw.csv"

def get_page(url):
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response.raise_for_status()
    response.encoding = "utf-8"
    time.sleep(0.1)
    return BeautifulSoup(response.text, "html.parser")


required_columns = [
    "title", "price", "rating", "stock",
    "category", "description", "upc", "availability"
]

# Reuse the CSV only if it contains valid book data.
raw = None

if os.path.exists(RAW_FILE):
    try:
        saved_data = pd.read_csv(RAW_FILE)

        if (
            not saved_data.empty
            and all(col in saved_data.columns for col in required_columns)
        ):
            raw = saved_data
            print("Loading existing book data...")
        else:
            print("Existing CSV is empty or invalid. Scraping again.")

    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        print("Could not read existing CSV. Scraping again.")

if raw is None:
    books = []

    for page in range(1, 51):
        page_url = urljoin(BASE, f"page-{page}.html")
        soup = get_page(page_url)

        items = soup.select("article.product_pod")

        if not items:
            raise RuntimeError(
                f"No books found on page {page}. "
                "Check the website response and URL."
            )

        for item in items:
            link = item.select_one("h3 a")
            price_element = item.select_one("p.price_color")
            rating_element = item.select_one("p.star-rating")
            stock_element = item.select_one("p.availability")

            if not all([
                link, price_element, rating_element, stock_element
            ]):
                continue

            book = {
                "title": link.get("title", ""),
                "price": price_element.get_text(strip=True),
                "rating": rating_element.get("class", ["", "Unknown"])[1],
                "stock": stock_element.get_text(" ", strip=True),
            }

            # Build the correct URL for each book's detail page.
            detail_url = urljoin(page_url, link["href"])
            detail = get_page(detail_url)

            breadcrumbs = detail.select("ul.breadcrumb li a")

            if len(breadcrumbs) < 3:
                continue

            book["category"] = breadcrumbs[2].get_text(strip=True)

            desc = detail.select_one("#product_description")
            book["description"] = (
                desc.find_next_sibling("p").get_text(strip=True)
                if desc and desc.find_next_sibling("p")
                else ""
            )

            rows = detail.select("table.table-striped tr")
            info = {
                row.th.get_text(strip=True): row.td.get_text(strip=True)
                for row in rows
                if row.th and row.td
            }

            book["upc"] = info.get("UPC", "")
            book["availability"] = info.get("Availability", "")

            books.append(book)

        print(f"Page {page} done — {len(books)} books collected")

    raw = pd.DataFrame(books, columns=required_columns)

    if raw.empty:
        raise RuntimeError(
            "No books were scraped. Check your internet connection, "
            "URLs, and the website response."
        )

    raw.to_csv(RAW_FILE, index=False)
    print("Raw book data saved.")

print("Books scraped:", len(raw))

if raw.empty or not all(col in raw.columns for col in required_columns):
    raise ValueError(
        "Book data is empty or missing required columns. "
        "Check data/books_raw.csv."
    )

# ---------------------------------------------------------------
# 2: CLEAN THE DATA
# ---------------------------------------------------------------

df = raw.drop_duplicates(subset="upc").copy()

# Convert prices to numbers.
df["price"] = pd.to_numeric(
    df["price"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
    errors="coerce"
)

# Convert star ratings to numbers.
stars = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["rating"].map(stars)

# Determine stock status and quantity.
df["availability"] = df["availability"].fillna("")

df["in_stock"] = df["availability"].str.contains(
    "In stock", case=False, na=False
)

df["stock_qty"] = (
    df["availability"]
    .str.extract(r"\((\d+)\s+available\)", expand=False)
    .astype(float)
    .fillna(0)
)

# Clean descriptions and calculate word counts.
df["description"] = df["description"].fillna("")

df["has_description"] = (
    df["description"] != ""
).astype(int)

df["desc_words"] = df["description"].str.split().str.len()
df["title_words"] = df["title"].fillna("").str.split().str.len()

# Remove rows with missing values needed for analysis.
df = df.dropna(subset=["price", "rating", "category"])

if len(df) < 2:
    raise ValueError(
        "Not enough valid book records for analysis. "
        "Delete data/books_raw.csv and run the script again."
    )

df = df[
    [
        "title", "category", "price", "rating",
        "in_stock", "stock_qty", "has_description",
        "desc_words", "title_words"
    ]
].copy()

df.to_csv("data/books_clean.csv", index=False)

print("\n--- Summary statistics ---")
print(df.describe().round(2))

# ---------------------------------------------------------------
# 3: VISUALISE THE DATA
# ---------------------------------------------------------------

# Price distribution.
plt.hist(df["price"], bins=30, color="steelblue")
plt.title("Price Distribution")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.savefig("figures/1_price_hist.png")
plt.close()

# Price by star rating.
groups = [
    df.loc[df["rating"] == rating, "price"]
    for rating in [1, 2, 3, 4, 5]
]

plt.boxplot(groups)
plt.xticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"])
plt.title("Price by Star Rating")
plt.xlabel("Stars")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.savefig("figures/2_price_by_rating.png")
plt.close()

# Average price for the ten largest categories.
top10 = df["category"].value_counts().head(10).index

avg = (
    df[df["category"].isin(top10)]
    .groupby("category")["price"]
    .mean()
    .sort_values()
)

avg.plot(kind="barh", color="steelblue")
plt.title("Average Price by Genre (10 Largest Categories)")
plt.xlabel("Average Price (£)")
plt.tight_layout()
plt.savefig("figures/3_price_by_genre.png")
plt.close()

# Price versus description length.
plt.scatter(df["desc_words"], df["price"], alpha=0.4)
plt.title("Price vs Description Length")
plt.xlabel("Words in Description")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.savefig("figures/4_price_vs_description.png")
plt.close()

print("\n--- Correlation with Price ---")
print(
    df[
        ["price", "rating", "stock_qty", "desc_words", "title_words"]
    ].corr()["price"].round(3)
)

# ---------------------------------------------------------------
# 4: LINEAR REGRESSION
# ---------------------------------------------------------------

counts = df["category"].value_counts()

df["genre"] = np.where(
    df["category"].map(counts) >= 15,
    df["category"],
    "Other"
)

X = df[
    [
        "rating", "stock_qty", "desc_words",
        "title_words", "has_description"
    ]
].astype(float)

# Add category indicator columns.
X = pd.concat(
    [
        X,
        pd.get_dummies(
            df["genre"],
            drop_first=True,
            dtype=float
        )
    ],
    axis=1
)

y = df["price"].to_numpy()

# Split the data into training and testing sets.
np.random.seed(42)

order = np.random.permutation(len(df))
cut = int(0.8 * len(df))

train = order[:cut]
test = order[cut:]

if len(train) == 0 or len(test) == 0:
    raise ValueError("Not enough records to split into training and test sets.")

# Add an intercept column and fit the linear regression model.
A = np.column_stack([np.ones(len(X)), X.to_numpy()])

coef, *_ = np.linalg.lstsq(
    A[train],
    y[train],
    rcond=None
)

# Predict prices for the test set.
pred = A[test] @ coef
actual = y[test]

# Calculate evaluation metrics.
mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred) ** 2))

total_variation = np.sum((actual - actual.mean()) ** 2)

r2 = (
    1 - np.sum((actual - pred) ** 2) / total_variation
    if total_variation != 0
    else float("nan")
)

base = y[train].mean()
base_mae = np.mean(np.abs(actual - base))

print("\n--- Model Results (Test Set) ---")
print(f"R-squared : {r2:.3f}")
print(f"MAE       : £{mae:.2f} (baseline: £{base_mae:.2f})")
print(f"RMSE      : £{rmse:.2f}")

# Display the most influential coefficients by absolute size.
names = ["intercept"] + list(X.columns)
result = pd.Series(coef, index=names).round(3)

print("\n--- Largest Regression Coefficients ---")
print(result.sort_values(key=abs, ascending=False).head(10))

# Plot predicted versus actual prices.
plt.scatter(actual, pred, alpha=0.5)
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    "r--"
)

plt.title("Predicted vs Actual Price")
plt.xlabel("Actual Price (£)")
plt.ylabel("Predicted Price (£)")
plt.tight_layout()
plt.savefig("figures/5_predicted_vs_actual.png")
plt.close()

print("\nDone! Cleaned data is in the 'data' folder.")
print("Graphs are saved in the 'figures' folder.")