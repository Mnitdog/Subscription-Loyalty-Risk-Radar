# **Subscription Loyalty Risk Radar**

<p align="center">

  <img src="https://img.shields.io/badge/Type-ML%20System-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Modeling-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-Data--Processing-yellow?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/ML%20Pipeline-Subscription%20%7C%20Frequency%20Models-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Explainability-Feature%20Importance-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Loyalty-Risk%20Scoring-critical?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Code%20Style-PEP8-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />

</p>

*An intelligence engine that transforms raw shopping behavior into subscription insights, frequency predictions, loyalty scoring, and scenario simulation, designed for teams that want to understand not only what customers do, but why they behave the way they do.*

---

# **Overview**

Retail loyalty is not a single action, it is a *behavioral signature* that emerges from repeated decisions: purchasing rhythms, shipping preferences, discount sensitivity, past experiences, and long-term commitment tendencies.

Yet most companies reduce loyalty to naive metrics like “number of purchases” or “subscription status.”
This leads to simplistic marketing decisions and predictable churn.

**Subscription-Loyalty-Risk-Radar** takes a more scientific view:

* Loyalty is multi-dimensional
* Behavior must be quantified
* Predictions must be explainable
* Insights must be actionable

This project builds a **full-stack ML system** that:

### 1. Predicts **subscription probability**

Who is likely to subscribe? Who is unlikely? Why?

### 2. Models **purchase frequency**

How often will a customer buy? What is their behavioral “intensity score”?

### 3. Creates a unified **Loyalty Risk Score (0–100)**

A single interpretable metric combining short-term behavior + long-term intent.

### 4. Provides explainability for each score

Which features raised or lowered loyalty? What factors shape behavior?

### 5. Simulates **what-if scenarios**

What happens if you offer a discount? Change shipping speed? Add a promo?

### 6. Visualizes everything in an **interactive dashboard**

A complete customer intelligence interface powered by Streamlit.

---

# **Why This Project Exists**

(*A business narrative + a data science narrative*)

## The Business Problem

E-commerce teams struggle with questions like:

* “Which customers are slipping away?”
* “Who should we target with retention offers?”
* “Which segments are discount-driven?”
* “What would increase subscription adoption?”
* “Who buys weekly vs monthly vs annually, and why?”

And crucially:

> **“Which levers actually change customer behavior?”**
> (not which ones we *think* do)

Traditional dashboards fail because they answer *what happened*, but not *what will happen* or *why it will happen*.

This project fills that gap.

---

## The Data Science Problem

Most ML pipelines try to predict a single target.
But loyalty is *not* a single target, it is the interaction of at least two dimensions:

### **1. Long-term commitment signals → subscription intention**

This reflects trust, brand fit, and willingness to commit.

### **2. Short-term behavioral intensity → purchase frequency**

This reflects habits, timing, product needs, lifestyle cycles.

These two dimensions **do not always correlate**, which is why a single model is insufficient.

A customer may:

* Buy frequently but never subscribe
* Buy rarely but have high subscription tendency
* Buy seasonally yet be highly loyal
* Buy many times but be price-sensitive and churn-prone

To model loyalty correctly, we must model:

* **Intent**
* **Behavior**
* **Consistency**
* **Sensitivity**
* **Predictability**

This system captures all of them.

---

# **How a Data Scientist Thinks About Loyalty**

(*Core design philosophy*)

### **Loyalty is not an outcome, it is an evolving probability distribution.**

We build models not to label customers but to **approximate their latent state**.

### **Prediction is only step 1, interpretation is step 2.**

A high churn score is meaningless unless we know the reason.

### **The system must generate strategy.**

Knowing someone is “at risk” is not enough.
We need to answer:

* What lever would improve their loyalty?
* What scenario reduces their risk most?
* How does discount sensitivity differ across personas?

### **Human + Machine collaboration**

This tool is not meant to replace analysts, it amplifies them.

---

# **System Architecture**

Below is a conceptual high-level diagram (not code-specific):

```
        ┌────────────────────────────────────┐
        │        Raw Shopping Dataset        │
        └────────────────────────────────────┘
                          │
                          ▼
        ┌────────────────────────────────────┐
        │      Data Cleaning & Normalization │
        └────────────────────────────────────┘
                          │
                          ▼
        ┌────────────────────────────────────┐
        │   Feature Engineering & Encoding   │
        └────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
 ┌────────────────────────┐    ┌──────────────────────────┐
 │   Subscription Model   │    │   Frequency Regression   │
 │ (Binary Classification)│    │ (Ordinal Behavior Score) │
 └────────────────────────┘    └──────────────────────────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
        ┌────────────────────────────────────┐
        │       Loyalty Scoring Engine       │
        │ (combine probability + frequency)  │
        └────────────────────────────────────┘
                          │
                          ▼
        ┌────────────────────────────────────┐
        │      Streamlit Intelligence UI     │
        └────────────────────────────────────┘
```

---

# **Dataset Signals Used**

The models leverage a mixture of:

### **Demographics**

* Age
* Gender
* Location

### **Purchasing Behavior**

* Purchase amount
* Previous purchases
* Frequency of purchases (target for frequency model)

### **Experience Signals**

* Review rating
* Shipping type
* Discount use
* Promo code use

### **Product Preference**

* Category
* Item purchased
* Color
* Size
* Season

Together, these features reflect both *identity* and *behavior*, crucial for modeling loyalty.

---

# **Modeling Strategy**

## Subscription Model

**Question:** *“If we removed friction, how likely is this customer to subscribe?”*

### Why Random Forest?

* Handles non-linear relationships (“young + winter + clothing discount = subscriber”)
* Robust to noise
* Performs well with mixed categorical + numeric data
* Avoids overfitting with minimal tuning

### What the model learns:

* Customers who buy frequently trend toward subscribing
* Promo usage may indicate value sensitivity
* Shipping preference indicates tolerance for speed vs. cost
* Location interacts with seasonality
* Certain product categories correlate with subscription behavior

---

## Frequency Model

**Question:** *“How strong is this customer’s purchasing rhythm?”*

The target is treated as an **ordinal variable**, converted to an intensity scale (1–7).

### Why a regressor (instead of classification)?

Because:

* The distance between categories matters
* Weekly ≠ Fortnightly ≠ Monthly
* Regression treats the output as a continuum
* Allows subtle differences between customers

It essentially measures **habit strength**.

---

## Loyalty Scoring Engine

We model loyalty as:

```
Loyalty = Intent (60%) + Behavior (40%)
```

Why?

* **Subscription intention** reflects *commitment*
* **Frequency score** reflects *habit strength*

Both matter, but intention is slightly more predictive long-term.

Then we compute:

### **Loyalty Index (0–1 scale)**

```
loyalty_index = 
      0.6 * p_subscribe 
    + 0.4 * (frequency_score / 7)
```

### **Loyalty Risk (0–100 scale)**

```
loyalty_risk = (1 - loyalty_index) * 100
```

High risk means:

* Low frequency + low subscription probability
* Inconsistent or seasonal buying pattern
* Price-sensitivity with low commitment
* Weak habit + friction sensitivity

---

# **Segment Intelligence (Why This Matters)**

Segment-level insights reveal patterns like:

* **Winter clothing buyers** may be high-frequency but low-subscriber
* **Cash users** may have sporadic behavior
* **Express shipping demand** might correlate with loyalty
* **Promo-heavy shoppers** may churn if discounts stop

These insights guide:

* Marketing personalization
* Pricing strategy
* Retention campaigns
* Seasonal promotions
* Subscription product design

---

# **Scenario Simulation**

This is one of the most powerful features.

You can modify a customer’s attributes to answer:

### *“If I change X, what would happen to loyalty?”*

Examples:

* Change shipping from “Standard” → “Express”
* Toggle “Discount Applied: Yes → No”
* Add a promo code
* Switch payment method

The system recomputes:

* New subscription probability
* New frequency score
* New loyalty risk
* And shows the delta for each metric

This helps teams test strategies *before* deploying them.

---

# **Explainability**

Marketing and product teams care about:

* “Why did the model say this customer is at risk?”
* “What drives loyalty in this segment?”

Explainability provides:

### **Global feature importance**

What factors matter most overall?

### **Local (per-customer) explanations**

Which features increased or decreased:

* Intent
* Frequency
* Loyalty

This turns predictions into **stories**:

* “This customer buys weekly but rarely uses discounts, high loyalty.”
* “This customer buys only in winter and always uses promos, seasonal but price-sensitive.”
* “This customer prefers express shipping and leaves high reviews, strong subscription potential.”

Now the model is not a black box.
It is a diagnostic tool.

---

# **Quickstart**

```bash
pip install -r requirements.txt
```

```bash
python -m src.cli prepare-data
python -m src.cli train-all
python -m src.cli evaluate
python -m src.cli score-customers --output data/processed/scored.parquet
streamlit run app/app.py
```

---

# **Future Enhancements**

### Machine Learning

* Replace RandomForest with **LightGBM** for better performance
* Hyperparameter optimization (Optuna)
* Add **ordinal regression** for frequency
* Add **seasonally aware** models

### Analytics

* Persona clustering (KMeans + PCA/UMAP)
* Retention funnel modeling
* Abandonment probability model
* Price elasticity modeling

### Dashboard UX

* Animated cohort transitions
* Customer “journey cards”
* Auto-generated retention recommendations

### Engineering

* FastAPI backend for scoring
* Docker containerization
* Full cloud deployment
* Automated monitoring + drift detection

---

# **Final Thoughts**

**Subscription-Loyalty-Risk-Radar** is more than an ML pipeline.
It is a **framework for understanding customer behavior**, built with:

* Mathematical clarity
* Business intuition
* System-level thinking
* Explainability
* Actionability

It shows how a data scientist:

* Designs multi-model systems
* Thinks about latent customer states
* Blends prediction with reasoning
* Turns algorithms into decisions
* Makes machine learning useful

This is not just a model,
**it is a loyalty intelligence engine.**
