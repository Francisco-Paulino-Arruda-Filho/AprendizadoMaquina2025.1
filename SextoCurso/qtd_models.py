learning_rate = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2]
max_depth = [4,6,8,10,12,14,16,18, 20]
subsample = [0.4, 0.6, 0.7, 0.8, 0.9]
max_features = ['auto', 'sqrt', 'log2']
result_list = []

for lr in learning_rate:
    for md in max_depth:
        for ss in subsample:
            for mf in max_features:
                result_list.append((lr, md, ss, mf))

print("Total combinations:", len(result_list))