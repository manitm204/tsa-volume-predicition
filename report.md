# TSA Volume Forecasting – Beginner Applicaiton - Manit Mahajan

## 1. Methodology

### Model Selection and Rationale

In previous time-series work, I focused on stock market prediction using both LSTM and Transformer-based models. LSTMs are effective at modeling sequential dependencies, while Transformers leverage attention mechanisms to capture long-range temporal interactions. Although these deep learning models are powerful, they are most effective when working with large-scale raw sequential data.

For this TSA volume forecasting task, a tree-based model was more appropriate. The dataset is structured and benefits heavily from engineered features such as calendar effects, lags, and rolling statistics. Tree-based models excel in tabular settings because they naturally capture nonlinear feature interactions without requiring extensive feature scaling or architecture tuning.

I selected XGBoost over alternatives such as Random Forest because XGBoost includes built-in regularization, better handling of feature interactions, and more efficient boosting mechanics. Compared to a single decision tree or standard Random Forest, XGBoost typically provides stronger generalization and better performance for structured regression problems.

---

### Model Configuration and Key Improvements

The final model used:

* `n_estimators = 3000`
* `learning_rate = 0.03`
* `max_depth = 4`
* `subsample = 0.8`
* `colsample_bytree = 0.8`
* `reg_alpha = 0.2`
* `reg_lambda = 1.5`

I explored hyperparameter adjustments carefully rather than over-optimizing for this single dataset. One of the most impactful changes was reducing tree depth from 6 to 4. This reduced MAE from approximately 115,000 to 105,000, nearly an 8–9% improvement. Shallower trees generalized better because the model already contained strong lag and rolling features. Deeper trees tended to overfit short-term noise, while depth=4 forced the model to learn broader structural relationships.

Another major improvement came from expanding the lag structure. Initially, I included lags at 1, 7, 14, 21, 28, 56, and 365 days. However, this missed very short-term momentum. After adding 2-day and 3-day lags, MAE decreased substantially from around 104,000 to 95,000. This demonstrated that short-term demand dynamics play a meaningful role in TSA volume.

To better capture extreme travel days, I implemented percentile-based sample weighting. Testing different percentile cutoffs revealed that emphasizing the top 90–95% of volume days significantly improved peak performance without harming stability on regular days.

---

## 2. Core Feature Engineering

Calendar features formed the structural foundation of the model. TSA volume is highly dependent on weekly and yearly seasonality, so features such as day of week, week of year, month, quarter, and day of year were included. Cyclical encodings using sine and cosine transformations allowed the model to treat seasonal cycles smoothly rather than as hard discontinuities.

Holiday-related features were also critical. Federal holiday indicators, along with days-to-holiday and days-since-holiday variables, allowed the model to anticipate demand shifts around travel periods. After observing underprediction during Thanksgiving and Christmas, I added more explicit holiday ramp features, including days to Thanksgiving (clipped to ±21 days), Thanksgiving week flags, and equivalent features for Christmas. These additions improved the model’s ability to anticipate holiday surges rather than react to them.

Lag, rolling mean, rolling standard deviation, and exponentially weighted moving averages were essential in capturing autoregressive structure. These features allowed the tree model to “remember” past demand patterns, since trees do not inherently model temporal dependence. Expanding the lag set to include very short-term lags proved to be one of the most impactful changes in the entire modeling process.

Weather data was sourced from Open-Meteo using major U.S. hub cities. Early attempts using simple averages (temperature, precipitation, windspeed) were too basic and did not meaningfully improve predictions. TSA volume is most affected by disruptive weather, not mild daily variation. Therefore, I engineered features representing widespread high wind, heavy snow, extreme cold, maximum gust speeds, and a composite weather disruption score. A binary severe-weather flag was also created to detect days likely associated with cancellations. These engineered features better captured major winter storm effects that caused large traffic drops.

---

## 3. Data Analysis and Handling

The training and test datasets were combined temporarily to ensure consistent chronological feature engineering. This was necessary because lag and rolling features must be generated strictly in time order to avoid leakage. After feature construction, the data was split back into training and test sets.

The target variable (Volume) was log-transformed using log1p() to stabilize variance and reduce the impact of extreme values during training. After prediction, results were transformed back using expm1() to return to the original scale.

Missing data handling included linear interpolation for small weather gaps, natural exclusion of rows with insufficient lag history, and placeholder values for holiday distance variables when no prior or future holiday existed. These steps ensured that feature construction remained stable and leakage-free.

Model evaluation relied on MAE, RMSE, and MAPE. MAE was prioritized because it directly reflects average passenger error, making it operationally intuitive. RMSE was monitored to ensure that extreme misses were not dominating performance, and MAPE provided scale-relative context.

---

## 4. Learnings and Challenges

The most challenging aspect of building this baseline model was accurately handling peak travel days. Early versions of the model captured weekly seasonality and overall trend well but consistently underpredicted major holiday surges, particularly Thanksgiving and Christmas. Addressing this required a combination of structural feature improvements and loss-shaping techniques. Introducing percentile-based sample weighting significantly improved the model’s ability to handle extreme demand days. Expanding lag structure to include short-term dependencies further enhanced peak responsiveness.

Weather feature engineering also proved challenging. Initial implementations using simple averages did not meaningfully impact predictions because mild weather variation does not strongly affect national TSA volume. The key insight was that the model needed to detect severe, disruptive events, such as widespread winter storms. Engineering a stricter severe-weather indicator improved performance during large late-January traffic drops. However, capturing extreme operational disruptions remains difficult without more granular cancellation or airport-level throughput data.

Another complexity is structural breaks such as COVID. The model learns historical patterns but does not inherently understand regime shifts. Incorporating regime indicators or macroeconomic context could improve robustness in future iterations.

---

## 5. Additional Insights and Future Work

The current implementation predicts next-day TSA volume. However, for applications such as Kalshi trading markets, predicting weekly totals would be more directly relevant. Transitioning to multi-day forecasting introduces additional complexity because future weather must be forecasted rather than observed, increasing uncertainty.

Future improvements could include building direct multi-horizon models, generating prediction intervals to quantify uncertainty, incorporating regime detection features, and integrating macroeconomic indicators such as gas prices, unemployment rates, consumer confidence, or Google search trends related to air travel. Additionally, trading applications would require not only a point forecast but also a confidence measure to determine when deviations from market-implied expectations are large enough to justify taking a position. High-volatility weeks, such as major storm forecasts or holiday travel periods, would likely require special handling due to increased uncertainty.
