class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')

# Extract unique identifiers (CSN) from the original DataFrame
unique_csns = final.index.get_level_values('CSN').unique()

# Split the unique CSNs into train, validation, and test sets
csn_test = np.random.choice(unique_csns, size=int(0.2 * len(unique_csns)), replace=False)
csn_remaining = np.setdiff1d(unique_csns, csn_test)
csn_val = np.random.choice(csn_remaining, size=int(0.25 * len(csn_remaining)), replace=False)
csn_train = np.setdiff1d(csn_remaining, csn_val)

# Subset the original DataFrame based on the split CSNs
df_test = final.loc[csn_test]
df_val = final.loc[csn_val]
df_train = final.loc[csn_train]

# Verify the lengths of the splits
print("Length of df_test:", len(df_test))
print("Length of df_val:", len(df_val))
print("Length of df_train:", len(df_train))

# Extract the indices for each split
test_index = np.where(unique_csns.isin(csn_test))[0]
val_index = np.where(unique_csns.isin(csn_val))[0]
train_index = np.where(unique_csns.isin(csn_train))[0]

# Aggregate or select one row per CSN
df_train_unique = df_train.groupby(level='CSN').first().reset_index()
df_val_unique = df_val.groupby(level='CSN').first().reset_index()
df_test_unique = df_test.groupby(level='CSN').first().reset_index()

# Extract time-to-event and event label for unique CSN
num_durations = 10
labtrans = LabTransform(num_durations)
get_target = lambda df: (df['ED_LOS'].values, df['Outcome'].values)
y_train_surv = labtrans.fit_transform(*get_target(df_train_unique))
y_val_surv = labtrans.transform(*get_target(df_val_unique))

# We don't need to transform the test labels
durations_test, events_test = (df_test_unique['ED_LOS'].values, df_test_unique['Outcome'].values)

out_features = labtrans.out_features  # how many discrete time points to predict for (10 here)
cuts = labtrans.cuts

# Verify the lengths of the transformed data
print("Length of y_train_surv:", len(y_train_surv[0]))
print("Length of y_val_surv:", len(y_val_surv[0]))

df_train.shape, df_val.shape, df_test.shape, df_train_unique.shape, df_val_unique.shape, df_test_unique.shape

final.drop(columns=['ED_LOS', 'Outcome'], inplace=True)

for column_index, column_name in enumerate(final.columns):
  print(f"Index: {column_index}, Column Name: {column_name}")

def split_sequence_optimized(dataframe, n_steps):
    # Get unique patient IDs
    patient_ids = dataframe.index.get_level_values(0).unique()
    num_patients = len(patient_ids)

    # Pre-allocate the LSTM input array with NaNs
    lstm_input = np.full((num_patients, n_steps, dataframe.shape[1]), np.nan)

    # Iterate through each patient
    for i, patient_id in enumerate(patient_ids):
        # Extract data for the current patient
        patient_data = dataframe.loc[patient_id].values

        # Calculate the starting index for the sequence
        start_index = max(0, patient_data.shape[0] - n_steps)

        # Assign the sequence to the LSTM input array
        lstm_input[i, -patient_data.shape[0]:, :] = patient_data[start_index:]

    # Replace NaNs with the last value in the sequence
    for i in range(lstm_input.shape[0]):  # Iterate over patients
        for j in range(lstm_input.shape[2]):  # Iterate over features
            # Find the last non-NaN value in the sequence
            last_valid_index = np.where(~np.isnan(lstm_input[i, :, j]))[0]
            if len(last_valid_index) > 0:
                last_valid_value = lstm_input[i, last_valid_index[-1], j]
                # Replace NaNs with the last valid value
                lstm_input[i, :, j] = np.where(
                    np.isnan(lstm_input[i, :, j]),
                    last_valid_value,
                    lstm_input[i, :, j]
                )

    return lstm_input

# Extract data into timeseries format with 24 timesteps
timeseries_lstm_input = split_sequence_optimized(dataframe=final, n_steps=24)

timeseries_lstm_input.shape, np.isnan(timeseries_lstm_input).any()

len(test_index), len(train_index), len(val_index)

timeseries_lstm_input_train = timeseries_lstm_input[train_index, :, :]
timeseries_lstm_input_val = timeseries_lstm_input[val_index, :, :]
timeseries_lstm_input_test = timeseries_lstm_input[test_index, :, :]

# Quantile transform the features across the population for each timestep
scalers = {}
for i in range(timeseries_lstm_input_train.shape[1]):
    scalers[i] = QuantileTransformer(random_state=0)
    timeseries_lstm_input_train[:, i, :38] = scalers[i].fit_transform(timeseries_lstm_input_train[:, i, :38])

for i in range(timeseries_lstm_input_val.shape[1]):
    timeseries_lstm_input_val[:, i, :38] = scalers[i].transform(timeseries_lstm_input_val[:, i, :38])

for i in range(timeseries_lstm_input_test.shape[1]):
    timeseries_lstm_input_test[:, i, :38] = scalers[i].transform(timeseries_lstm_input_test[:, i, :38])

x_train = timeseries_lstm_input_train.astype('float32')
x_val = timeseries_lstm_input_val.astype('float32')
x_test = timeseries_lstm_input_test.astype('float32')

x_train.shape, x_test.shape, durations_test.shape, y_train_surv[0].shape

out_features = labtrans.out_features # how many discrete time points to predict for (10 here)
cuts = labtrans.cuts