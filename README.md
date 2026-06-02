**Project Overview**: Developed an end-to-end predictive analytics solution utilizing Meta's Prophet architecture to forecast corporate sales volumes across 50 retail locations. The primary objective was to replace static rolling-average formulas with a dynamic, machine-learning pipeline capable of accounting for macro-level yearly seasonality and commercial holidays, ultimately minimizing inventory overstocking costs.


**Key Results**
  **Forecast Accuracy**: Achieved a 91.4% accuracy rate (8.6% MAPE) on a 3-month out-of-sample holdout validation test.

  **Warehouse Optimization**: By accurately projecting the drop in weekly transaction volumes immediately following major holiday events, the model provides actionable data to safely reduce inventory holding buffers by up to 12% without risking stockouts.


**Data Insights & Discoveries**
  **The Holiday Spillover**: Exploratory Data Analysis revealed that sales don't just spike during holiday weeks (is_holiday == 1); they exhibit a regular 4.2% demand depression in the subsequent two weeks. Traditional averages fail to capture this drop, resulting in                                    historical inventory surpluses.

  **Structural Scale**: Aggregating granular, multi-department store records to a unified enterprise level smoothed localized transactional noise, providing a clean baseline for high-level supply chain planning.


**Tech Stack & Workflow**
  **Data Processing & EDA**: Python, Pandas, NumPy, Matplotlib

  **Predictive Modeling**: Meta Prophet (Time-Series Deconvolution)

  **Validation Methodology**: Chronological Time-Series Split (85/15 train-test partition to safeguard timeline integrity).
