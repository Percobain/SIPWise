import pandas as pd

# Load both BTC-INR datasets (update with your actual file names)
df1 = pd.read_csv('BTC-INR.csv')  # Older dataset (2015–2020)
df2 = pd.read_csv('BTC_INR_Historical Data.csv')  # Newer dataset (2019–2021)

# Convert 'Date' column to datetime
df1['Date'] = pd.to_datetime(df1['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])

# Set 'Date' as index
df1.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)

# Sort by date (just in case)
df1.sort_index(inplace=True)
df2.sort_index(inplace=True)

# Merge: df2 values override df1 in overlapping dates
merged = df1.combine_first(df2)
merged.update(df2)

# Reset index to save
merged.reset_index(inplace=True)

# Save merged CSV
merged.to_csv('btc_inr_final.csv', index=False)

print(f"Merged dataset saved as btc_inr_final.csv with {len(merged)} rows.")
