"""# Vital signs"""

time_series_vitals = pd.read_csv('numerics.csv')

# Convert Result_time to datetime, handling out-of-bounds dates
time_series_vitals['Time'] = pd.to_datetime(time_series_vitals['Time'], errors='coerce')

# Remove rows with NaT (Not a Time) in Result_time, which were created due to out-of-bounds dates
time_series_vitals = time_series_vitals[time_series_vitals['Time'].notna()]

# Format Result_time as 'YYYY-MM-DD HH:MM:SS'
time_series_vitals['Time'] = time_series_vitals['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Pivot the DataFrame
time_series_vitals = time_series_vitals.pivot_table(
    index=['CSN', 'Time'],  # Index: CSN and Result_time
    columns='Measure',      # Columns: Unique Component_name entries
    values='Value',      # Values: Component_value (or another column like Component_result)
    aggfunc='first'                # Handle duplicates: Take the first occurrence
)

# Convert problematic columns to numeric before calculating the mean
for column in time_series_vitals.columns:
    if column != 'CSN' and column != 'Time':  # Exclude non-numeric columns if present
        time_series_vitals[column] = pd.to_numeric(time_series_vitals[column], errors='coerce')

time_series_vitals.reset_index(inplace=True)
time_series_vitals['Time'] = pd.to_datetime(time_series_vitals['Time'], errors='coerce')
time_series_vitals.set_index(['CSN', 'Time'], inplace=True)
# Time shift so that the starting point for each sample is time = 0
time_series_vitals.reset_index(level=1, inplace=True)
minimum_shifts = time_series_vitals.groupby('CSN')['Time'].min()
time_series_vitals = time_series_vitals.merge(minimum_shifts, left_index=True, right_index=True)
time_series_vitals['Time'] = time_series_vitals['Time_x'] - time_series_vitals['Time_y']
time_series_vitals.drop(columns=['Time_x', 'Time_y'], inplace=True)
time_series_vitals.set_index(['Time'], append=True, inplace=True)

# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_vitals.index.get_level_values('Time')  # Get Result_time from the index

time_series_vitals = time_series_vitals.groupby(level=[0, 1]).mean()

# Round up the time-stamps to the next hour
time_series_vitals.reset_index(level=1, inplace=True)
time_series_vitals.Time = time_series_vitals.Time.dt.ceil(freq='h')
time_series_vitals.Time = pd.to_timedelta(time_series_vitals.Time, unit='T')
time_series_vitals.set_index('Time', append=True, inplace=True)
time_series_vitals.reset_index(level=0, inplace=True)

time_series_vitals = time_series_vitals.groupby('CSN').resample('h', closed='right', label='right').mean().drop(columns='CSN')

time_series_vitals.reset_index(level=1, inplace=True)
time_series_vitals = time_series_vitals[time_series_vitals['Time'].notna()]
time_series_vitals.Time = pd.to_timedelta(time_series_vitals.Time, errors='coerce')

time_series_vitals.update(time_series_vitals.groupby(level=0).ffill())
time_series_vitals.update(time_series_vitals.groupby(level=0).bfill())

time_series_vitals.reset_index(inplace=True)
time_series_vitals.set_index(['CSN', 'Time'], inplace=True)

# Keep columns with less than 85% missing values
threshold = 0.85
time_series_vitals = time_series_vitals.loc[:, time_series_vitals.isnull().sum() / len(time_series_vitals) < threshold]

# Drop missing values samples
missing_samples = time_series_vitals[time_series_vitals.isnull().any(axis=1)].index.get_level_values(0).tolist()
time_series_vitals.drop(missing_samples, level=0, axis=0, inplace=True)

# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_vitals.index.get_level_values('Time')  # Get Result_time from the index

# Calculate total hours from the TimedeltaIndex
result_time_hours = result_time_series.total_seconds() / 3600  # Get total seconds and convert to hours

time_series_labs.index = time_series_labs.index.rename('Time', level='Result_time')

time_series = time_series_vitals.merge(time_series_labs, left_index=True, right_index=True)

visits.drop(['MRN'], axis=1, inplace=True)

visits.set_index('CSN', inplace=True)

visits = visits[~visits.index.duplicated(keep='first')]

final = time_series.merge(visits, left_index=True, right_index=True)

# Drop missing values samples
missing_samples = final[final.isnull().any(axis=1)].index.get_level_values(0).tolist()
final.drop(missing_samples, level=0, axis=0, inplace=True)

# Convert to numeric, forcing errors to NaN
final['Age'] = pd.to_numeric(final['Age'], errors='coerce')

# Check for negative or unrealistic values
invalid_ages = final[(final['Age'] < 0) | (final['Age'] > 120)]
