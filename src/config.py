from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw" / "shopping_behavior.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ID_COLUMN = "Customer ID"

TARGET_SUBSCRIPTION = "Subscription Status"
TARGET_FREQUENCY = "Frequency of Purchases"

NUMERIC_FEATURES = [
    "Age",
    "Purchase Amount (USD)",
    "Previous Purchases",
    "Review Rating",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Item Purchased",
    "Category",
    "Location",
    "Size",
    "Color",
    "Season",
    "Shipping Type",
    "Discount Applied",
    "Promo Code Used",
    "Payment Method",
]

FREQUENCY_MAPPING = {
    "Annually": 1,
    "Every 3 Months": 2,
    "Quarterly": 3,
    "Monthly": 4,
    "Fortnightly": 5,
    "Bi-Weekly": 6,
    "Weekly": 7,
}

# Build reverse mapping for convenience
FREQUENCY_MAPPING_INV = {v: k for k, v in FREQUENCY_MAPPING.items()}
