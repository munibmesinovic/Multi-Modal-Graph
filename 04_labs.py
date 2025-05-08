"""# Labs"""

time_series_labs = pd.read_csv('labs.csv')

# Convert Result_time to datetime, handling out-of-bounds dates
time_series_labs['Result_time'] = pd.to_datetime(time_series_labs['Result_time'], errors='coerce')

# Remove rows with NaT (Not a Time) in Result_time, which were created due to out-of-bounds dates
time_series_labs = time_series_labs[time_series_labs['Result_time'].notna()]

# Format Result_time as 'YYYY-MM-DD HH:MM:SS'
time_series_labs['Result_time'] = time_series_labs['Result_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Pivot the DataFrame
time_series_labs = time_series_labs.pivot_table(
    index=['CSN', 'Result_time'],  # Index: CSN and Result_time
    columns='Component_name',      # Columns: Unique Component_name entries
    values='Component_value',      # Values: Component_value (or another column like Component_result)
    aggfunc='first'                # Handle duplicates: Take the first occurrence
)

# Keep columns with less than 90% missing values
threshold = 0.85
time_series_labs = time_series_labs.loc[:, time_series_labs.isnull().sum() / len(time_series_labs) < threshold]

# Convert problematic columns to numeric before calculating the mean
for column in time_series_labs.columns:
    if column != 'CSN' and column != 'Result_time':  # Exclude non-numeric columns if present
        time_series_labs[column] = pd.to_numeric(time_series_labs[column], errors='coerce')

time_series_labs.reset_index(inplace=True)
time_series_labs['Result_time'] = pd.to_datetime(time_series_labs['Result_time'], errors='coerce')
time_series_labs.set_index(['CSN', 'Result_time'], inplace=True)
# Time shift so that the starting point for each sample is time = 0
time_series_labs.reset_index(level=1, inplace=True)
minimum_shifts = time_series_labs.groupby('CSN')['Result_time'].min()
# minimum_shifts = timeseries_summary.time.min(level=0)
time_series_labs = time_series_labs.merge(minimum_shifts, left_index=True, right_index=True)
time_series_labs['Result_time'] = time_series_labs['Result_time_x'] - time_series_labs['Result_time_y']
time_series_labs.drop(columns=['Result_time_x', 'Result_time_y'], inplace=True)
time_series_labs.set_index(['Result_time'], append=True, inplace=True)

# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_labs.index.get_level_values('Result_time')  # Get Result_time from the index

# Calculate total hours from the TimedeltaIndex
result_time_hours = result_time_series.total_seconds() / 3600  # Get total seconds and convert to hours

# For plotting the distribution in days:
plt.figure(figsize=(10, 6))
plt.hist(result_time_hours, bins=30)

plt.xlabel('Result Time (Hours)')
plt.ylabel('Frequency')
plt.title('Distribution of Result Time')
plt.show()

# For descriptive statistics:
print(pd.Series(result_time_hours).describe())

# # take the mean of any duplicate index entries for unstacking
# time_series_labs = time_series_labs.groupby(level=[0, 1]).mean()

# # Round up the time-stamps to the next day
# time_series_labs.reset_index(level=1, inplace=True)
# time_series_labs['Result_time'] = pd.to_datetime(time_series_labs['Result_time'], errors='coerce')

# # Time shift so that the starting point for each sample is time = 0
# minimum_shifts = time_series_labs.groupby('CSN')['Result_time'].min()
# time_series_labs = time_series_labs.merge(minimum_shifts, left_index=True, right_index=True)
# time_series_labs['Result_time'] = time_series_labs['Result_time_x'] - time_series_labs['Result_time_y']
# time_series_labs.drop(columns=['Result_time_x', 'Result_time_y'], inplace=True)
# time_series_labs.set_index(['Result_time'], append=True, inplace=True)

# # # Calculate timedelta from a reference date (e.g., minimum date)
# # reference_date = time_series_labs['Result_time'].min()  # Or a specific date if needed
# # time_series_labs['Result_time'] = time_series_labs['Result_time'] - reference_date

#Proceed with other operations like ceil, set_index, etc.

time_series_labs = time_series_labs.groupby(level=[0, 1]).mean()

# Round up the time-stamps to the next hour
time_series_labs.reset_index(level=1, inplace=True)
time_series_labs.Result_time = time_series_labs.Result_time.dt.ceil(freq='h')
time_series_labs.Result_time = pd.to_timedelta(time_series_labs.Result_time, unit='T')
time_series_labs.set_index('Result_time', append=True, inplace=True)
time_series_labs.reset_index(level=0, inplace=True)

time_series_labs = time_series_labs.groupby('CSN').resample('h', closed='right', label='right').mean().drop(columns='CSN')

time_series_labs.reset_index(level=1, inplace=True)
time_series_labs = time_series_labs[time_series_labs['Result_time'].notna()]
time_series_labs.Result_time = pd.to_timedelta(time_series_labs.Result_time, errors='coerce')

time_series_labs.update(time_series_labs.groupby(level=0).ffill())
time_series_labs.update(time_series_labs.groupby(level=0).bfill())

len(pd.unique(time_series_labs.index.get_level_values(0)))

time_series_labs.reset_index(inplace=True)
time_series_labs.set_index(['CSN', 'Result_time'], inplace=True)

# Drop missing values samples
missing_samples = time_series_labs[time_series_labs.isnull().any(axis=1)].index.get_level_values(0).tolist()
time_series_labs.drop(missing_samples, level=0, axis=0, inplace=True)

# Access the 'Result_time' column (which is now a timedelta)
result_time_series = time_series_labs.index.get_level_values('Result_time')  # Get Result_time from the index

# Calculate total hours from the TimedeltaIndex
result_time_hours = result_time_series.total_seconds() / 3600  # Get total seconds and convert to hours

# For descriptive statistics:
print(pd.Series(result_time_hours).describe())