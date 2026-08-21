from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeRegressor, plot_tree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "customers": "olist_customers_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "translation": "product_category_name_translation.csv",
    }
    return {name: pd.read_csv(data_dir / filename) for name, filename in files.items()}


def build_consolidated_dataset(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = pd.merge(frames["items"], frames["orders"], on="order_id", how="left")
    products = pd.merge(frames["products"], frames["translation"], on="product_category_name", how="left")
    merged = pd.merge(merged, products, on="product_id", how="left")
    merged = pd.merge(merged, frames["payments"], on="order_id", how="left")
    merged = pd.merge(merged, frames["customers"], on="customer_id", how="left")
    merged = pd.merge(merged, frames["sellers"], on="seller_id", how="left")
    merged = pd.merge(
        merged,
        frames["reviews"][["order_id", "review_score", "review_comment_message"]],
        on="order_id",
        how="left",
    )
    return merged


def run_association_rules(order_items: pd.DataFrame, sample_frac: float) -> pd.DataFrame:
    sampled = order_items.sample(frac=sample_frac, random_state=42)
    transactions = (
        sampled.groupby(["order_id", "product_id"])["product_id"]
        .count()
        .unstack()
        .fillna(0)
        .astype(bool)
    )
    itemsets = apriori(transactions, min_support=0.0005, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame(columns=["product_1", "product_2", "support", "confidence", "lift"])

    rules = association_rules(itemsets, metric="lift", min_threshold=1)
    result = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    result["product_1"] = result["antecedents"].apply(lambda x: ", ".join(map(str, sorted(x))))
    result["product_2"] = result["consequents"].apply(lambda x: ", ".join(map(str, sorted(x))))
    return result[["product_1", "product_2", "support", "confidence", "lift"]].sort_values("lift", ascending=False)


def run_sentiment_baseline(comments: pd.Series) -> pd.DataFrame:
    df_comments = comments.dropna().reset_index(drop=True).to_frame(name="review_comment_message")
    labeled = df_comments.head(16).copy()
    labeled["sentiment"] = [
        "positivo", "positivo", "positivo", "negativo",
        "negativo", "negativo", "positivo", "negativo",
        "positivo", "positivo", "negativo", "positivo",
        "negativo", "positivo", "positivo", "negativo",
    ]

    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(labeled["review_comment_message"])
    model = MultinomialNB()
    model.fit(x_train, labeled["sentiment"])

    transformed_all = vectorizer.transform(df_comments["review_comment_message"])
    predictions = model.predict(transformed_all)
    return df_comments.assign(sentiment=predictions)


def run_decision_tree(order_items: pd.DataFrame) -> tuple[DecisionTreeRegressor, float]:
    x = order_items[["price", "freight_value"]]
    y = order_items["order_item_id"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(x_train, y_train)
    return model, model.score(x_test, y_test)


def save_tree_figure(model: DecisionTreeRegressor) -> None:
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=["price", "freight_value"], filled=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "decision_tree.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Olist data mining workflows.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample-frac", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    frames = load_csvs(Path(args.data_dir))
    consolidated = build_consolidated_dataset(frames)
    consolidated.to_csv(OUTPUTS_DIR / "consolidated_dataset.csv", index=False)

    consolidated.describe(include="all").transpose().to_csv(OUTPUTS_DIR / "dataset_profile.csv")
    run_association_rules(frames["items"], args.sample_frac).to_csv(OUTPUTS_DIR / "association_rules.csv", index=False)
    run_sentiment_baseline(consolidated["review_comment_message"]).to_csv(OUTPUTS_DIR / "sentiment_predictions.csv", index=False)

    tree_model, tree_score = run_decision_tree(frames["items"])
    save_tree_figure(tree_model)
    (OUTPUTS_DIR / "decision_tree_score.txt").write_text(f"R2 score: {tree_score:.6f}\n", encoding="utf-8")

    print(f"Decision tree score: {tree_score:.6f}")
    print(f"Outputs saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
