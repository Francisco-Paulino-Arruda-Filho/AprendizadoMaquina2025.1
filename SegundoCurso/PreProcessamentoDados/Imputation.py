from sklearn.impute import SimpleImputer

X_cats = music_dummies.drop("genre", axis=1).values
X_nums = music_dummies.drop(["genre", "popularity"], axis=1).values
y = music_dummies["genre"].values
X_train_cats, X_test_cats, y_train_cats, y_test_cats = train_test_split(X_cats, y, test_size=0.2, random_state=42)
X_train_nums, X_test_nums, y_train_nums, y_test_nums = train_test_split(X_nums, y, test_size=0.2, random_state=42)
imp_cats = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cats.fit_transform(X_train_cats)
X_train_cat = imp_cats.transform(X_test_cats)
imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_nums)
X_test_num = imp_num.transform(X_test_nums)
X_train = np.append((X_train_cat, X_train_num), axis=1)
X_test = np.append((X_test_cat, X_test_num), axis=1)