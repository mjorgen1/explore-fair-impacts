def balance_between_group_label_ratio(x_data, y_data, label_ratio):
    set_size = len(y_data)
    num_samples_0 = int(set_size * race_ratio[0])
    num_samples_1 = int(set_size * race_ratio[1])

    num_0P = int(set_size_0 * label_ratio[1])
    num_1P = int(set_size_1 * label_ratio[1])
    num_0N = int(set_size_0 * label_ratio[0])
    num_1N = int(set_size_1 * label_ratio[0])

    idx_0N = np.where((x_data[:, 1] == 0) & (y_data == 0))[0]
    idx_1N = np.where((x_data[:, 1] == 1) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, 1] == 0) & (y_data == 1))[0]
    idx_1P = np.where((x_data[:, 1] == 1) & (y_data == 1))[0]


    if len(idx_0P) < num_0P:
        num_0P = len(idx_0P)
        num_0N = int(num_0P/label_ratio[1] * label_ratio[0])
    if len(idx_0N) < num_0N:
        num_0N = len(idx_0N)
        num_0P = int(num_0N/label_ratio[0] * label_ratio[1])
    if len(idx_1P) < num_1P:
        num_1P = len(idx_1P)
        num_1N = int(num_1P/label_ratio[1] * label_ratio[0])
    if len(idx_1N) < num_1N:
        num_1N = len(idx_1N)
        num_1P = int(num_1N/label_ratio[0] * label_ratio[1])

    if idx_0N <= idx_1N:
        idx_1N = idx_0N
    else:
        idx_0N = idx_1N
    if idx_0P <= idx_1P:
        idx_1P = idx_0P
    else:
        idx_0P = idx_1P


    idx_0N = idx_0N[:num_0N]
    idx_1N = idx_1N[:num_1N]
    idx_0P = idx_0P[:num_0P]
    idx_1P = idx_1P[:num_1P]

    idx = sorted(np.concatenate((idx_0N,idx_0P,idx_1N,idx_1P)))
    return x_data[idx,:], y_data[idx]

def adjust_demographic_ratio(x_data, y_data, race_ratio, train_set_bound = 1000000):

    if len(y_data) > train_set_bound:
        set_size = train_set_bound
    else:
        set_size = len(y_data)
    num_samples_0 = int(set_size * race_ratio[0])
    num_samples_1 = int(set_size * race_ratio[1])

    idx_0 = np.where(x_data[:, 1] == 0)[0]
    idx_1 = np.where(x_data[:, 1] == 1)[0]
    if len(idx_0) < num_samples_0:
        num_samples_0 = len(idx_0)
        num_samples_1 = int(num_samples_0/race_ratio[0] * race_ratio[1])
    elif len(idx_1) < num_samples_1:
        num_samples_1 = len(idx_1)
        num_samples_0 = int(num_samples_1/race_ratio[1] * race_ratio[0])

    idx_0 = idx_0[:num_samples_0]
    idx_1 = idx_1[:num_samples_1]
    idx = sorted(np.concatenate((idx_0,idx_1)))
    return x_data[idx,:], y_data[idx]

        if len(np.where(x_data[:, 1] == 0)[0]) > set_size_upper_bound * race_ratio[0]:
            set_size_0 = set_size_upper_bound * race_ratio[0]
        else:
            set_size_0 = len(np.where(x_data[:, 1] == 0)[0])

        if len(np.where(x_data[:, 1] == 1)[0]) > set_size_upper_bound * race_ratio[1]:
            set_size_1 = set_size_upper_bound * race_ratio[1]
        else:
            set_size_1 = len(np.where(x_data[:, 1] == 0)[0])
if len(np.where(x_data[:, 1] == 0)[0]) > test_set_bound * race_ratio[0]:
    set_size_0 = test_set_bound * race_ratio[0]
else:
    set_size_0 = len(np.where(x_data[:, 1] == 0)[0])

if len(np.where(x_data[:, 1] == 1)[0]) > test_set_bound * race_ratio[1]:
    set_size_1 = test_set_bound * race_ratio[1]
else:
    set_size_1 = len(np.where(x_data[:, 1] == 0)[0])

idx_0 = np.where(x_data[:, 1] == 0)[0]
idx_1 = np.where(x_data[:, 1] == 1)[0]
idx_0 = idx_0[:int(set_size_0)]
idx_1 = idx_1[:int(set_size_1)]
idx = sorted(np.concatenate((idx_0,idx_1)))
