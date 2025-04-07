from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

users["sub_enc_level"] = le.fit_transform(users["subscribed"])

print(users[["subscribed", "sub_enc_level"]])