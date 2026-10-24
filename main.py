"""House Price Prediction - predicts house prices with Linear Regression and Random Forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def make_dataset(n=1000, seed=42):
    """Generate a realistic synthetic housing dataset."""
    rng = np.random.default_rng(seed)
    area = rng.integers(500, 4000, n)            # square feet
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 50, n)                 # years
    distance = rng.uniform(1, 30, n)             # km from city centre
    price = (
        50_000 + area * 120 + bedrooms * 15_000 - age * 1_000 - distance * 3_000
        + rng.normal(0, 25_000, n)
    )
    return pd.DataFrame({"area": area, "bedrooms": bedrooms, "age": age,
                         "distance_km": distance, "price": price.round(-2)})


def main():
    df = make_dataset()
    print(df.head(), "\n")

    X, y = df.drop(columns="price"), df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"{name:18s}  R2 = {r2_score(y_test, pred):.3f}   MAE = {mean_absolute_error(y_test, pred):,.0f}")

    lr = models["Linear Regression"]
    print("\nWhat each feature adds to the price (linear model):")
    for feature, coef in zip(X.columns, lr.coef_):
        print(f"  {feature:12s} {coef:+,.0f}")

    new_house = pd.DataFrame([{"area": 2000, "bedrooms": 3, "age": 10, "distance_km": 8}])
    print(f"\nPredicted price for {new_house.iloc[0].to_dict()}: {lr.predict(new_house)[0]:,.0f}")


if __name__ == "__main__":
    main()
