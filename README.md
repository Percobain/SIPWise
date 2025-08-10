# Goal-Based SIP Investment Predictor using Financial Modeling + Machine Learning

This project helps individuals plan their **monthly SIP (Systematic Investment Plan)** contributions based on:

* Their **financial goal** (target amount),
* **Time horizon** (in years), and
* **Risk appetite** (Conservative, Balanced, Aggressive)

It uses **real historical data** from 2007–2021 of asset classes:

* **Bitcoin**
* **Gold**
* **Nifty50 (Indian equity index)**
* **Fixed Deposit (FD)** (assumed stable return)

Once we calculate how these asset classes behave, we simulate portfolio growth and generate **synthetic data** to train a Machine Learning model — helping anyone figure out how much they should invest monthly to reach their goal.

---

## Project Workflow (Step-by-Step)

---

### 1. Data Collection & Cleaning

* Merged overlapping historical CSVs for Bitcoin (`BTC-INR`) from Yahoo Finance.
* Aligned monthly closing prices for:

  * Bitcoin
  * Gold
  * Nifty50
* Fixed Deposit data was assumed to have a **constant return of 6% per annum**.

Result: A clean dataset of **monthly prices** (2007–2021)

---

### 2. Yearly Return Calculation

We **resampled monthly data to yearly**, using the last closing price of each year, and computed **annual returns**:

```**Annual Return (year)** = (Price_end - Price_start) / Price_start```

This gave us a dataframe of **yearly % returns** for each asset.

---

### 3. CAGR: Compound Annual Growth Rate

We computed the **average annual growth** assuming compounding — more realistic than average returns.

$$
\text{CAGR} = \left( \frac{\text{Final Value}}{\text{Initial Value}} \right)^{\frac{1}{\text{Years}}} - 1
$$

Result:

* Bitcoin CAGR: \~170%
* Gold CAGR: \~10%
* Nifty50 CAGR: \~6.3%

---

### 4. Volatility (Standard Deviation of Returns)

Measures the **risk** or **fluctuation** in returns. Higher volatility = more uncertainty.

$$
\sigma = \sqrt{ \frac{1}{N-1} \sum_{i=1}^N (R_i - \bar{R})^2 }
$$

---

### 5. Sharpe Ratio

Measures **risk-adjusted return** — how much excess return you're getting **per unit of risk**.

$$
\text{Sharpe Ratio} = \frac{\text{Return} - \text{Risk-Free Rate}}{\text{Volatility}}
$$

Assuming risk-free rate = 4%.

---

### 6. Risk Profiles (Asset Allocations)

We defined 3 sample profiles:

```python
risk_profiles = {
    'Conservative': {'FD': 0.60, 'Gold': 0.30, 'Nifty50': 0.10, 'Bitcoin': 0.00},
    'Balanced':     {'FD': 0.30, 'Gold': 0.30, 'Nifty50': 0.40, 'Bitcoin': 0.00},
    'Aggressive':   {'FD': 0.00, 'Gold': 0.20, 'Nifty50': 0.60, 'Bitcoin': 0.20}
}
```

---

### 7. Portfolio Return & Volatility (Realistic Modeling)

For each **risk profile**, we calculate:

#### Portfolio Return:

Weighted average of asset CAGRs:

$$
\text{Portfolio Return} = \sum_{i} w_i \cdot \text{CAGR}_i
$$

Where:

* $w_i$: weight of asset $i$
* $\text{CAGR}_i$: CAGR of asset $i$

#### Portfolio Volatility:

We don't just average volatilities — instead, we use the **covariance matrix** of asset returns and calculate true portfolio volatility:

$$
\sigma_p = \sqrt{ \mathbf{w}^T \cdot \Sigma \cdot \mathbf{w} }
$$

Where:

* $\mathbf{w}$: vector of weights
* $\Sigma$: covariance matrix of asset returns

We excluded FD from the covariance matrix since it has 0 volatility — its impact is scaled down appropriately.

---

### 8. Monte Carlo Simulation (Investment Growth)

To simulate **realistic portfolio growth**, we model **monthly SIP investments** over a given time horizon using random monthly returns:

#### Process:

* For each month, add monthly SIP
* Apply a **random return** drawn from a normal distribution:

     ```
     R_month ~ N(r / 12, σ / √12)
     ```
     Where:
     - `r` is the annual expected return (CAGR)
     - `σ` is the annual volatility
     - `R_month` is the simulated monthly return

* Repeat for all months and for **1000 simulations**
* Average the final portfolio values from all simulations

This gives us the **expected goal amount** for a given SIP, duration, and risk profile.

---

### 9. Generating Synthetic Data for ML

We flipped the simulation logic to **generate data** for machine learning:

#### Inputs:

* **Monthly SIP amount** (₹1000 to ₹50000)
* **Investment duration** (1–20 years)
* **Risk profile** (Conservative, Balanced, Aggressive)

For each combination:

* Simulate investment growth via Monte Carlo
* Record the **goal amount reached**

#### Outputs:

* $X =$ \[Goal Amount, Duration, Risk Profile (encoded)]
* $y =$ Monthly SIP

We generated **480 synthetic data points** to train a regression model.

---

### 10. Machine Learning Model

We trained a `RandomForestRegressor` model to **predict the required monthly SIP**, given:

* Desired goal amount
* Duration (years)
* Risk profile

#### Evaluation:

* **MSE:** \~12.9 lakhs²
* **R² Score:** \~0.95 → Good fit
* **Feature importance:**

  * Goal Amount: \~64%
  * Duration: \~30%
  * Risk Profile: \~6%

This lets users answer:

> "How much should I invest monthly to reach ₹X in Y years with Z risk tolerance?"
>

Model Accuracy:

The model achieved an R² Score of 0.948, indicating that ~95% of the variance in SIP prediction is explained by the inputs.

This level of accuracy is excellent for financial estimation models, especially when working with real market volatility.

---

### 11. Gradio UI

We built a clean, interactive web UI using **Gradio** where users can:

* Enter their **goal**, **duration**, and **risk profile**
* View predicted **SIP amount**
* See an **investment growth chart**
* Visualize their **asset allocation pie chart**
* Read an **investment summary** (CAGR, total return, volatility)

---

## Final Outputs

* A trained ML model that predicts SIP for any user input
* Financial engine using real market data + Monte Carlo simulation
* Risk-adjusted metrics: CAGR, Volatility, Sharpe Ratio
* Gradio-powered interactive investment planner

---

## Glossary of Financial Terms

| Term             | Meaning                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| **SIP**          | Systematic Investment Plan – monthly investing strategy                 |
| **CAGR**         | Compound Annual Growth Rate – smoothed annual return                    |
| **Volatility**   | Standard deviation of returns – measures investment risk                |
| **Sharpe Ratio** | Return per unit of risk (vs. a risk-free asset like FD)                 |
| **Covariance**   | Measures how two assets move together – used for portfolio optimization |
| **Monte Carlo**  | Simulation technique using random sampling to model uncertain outcomes  |

---

## Why This Project Matters

* Most SIP calculators **hardcode returns** and ignore real volatility.
* We use **historical data + statistical modeling** to provide realistic, personalized estimates.
* This builds financial literacy + planning confidence in investors, even if they're new.

---


