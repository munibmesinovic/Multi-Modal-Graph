np.save('durations_test_MCMED.npy', durations_test)
np.save('events_test_MCMED.npy', events_test)
np.save('out_features_MCMED.npy', out_features)
np.save('cuts_MCMED.npy', cuts)
np.save('x_train_MCMED.npy', x_train)
np.save('x_val_MCMED.npy', x_val)
np.save('x_test_MCMED.npy', x_test)

import pickle

pickle.dump(y_train_surv, open('y_train_surv_MCMED.p', 'wb'))
pickle.dump(y_val_surv, open('y_val_surv_MCMED.p', 'wb'))