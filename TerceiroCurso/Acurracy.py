from sklearn.model_selection import train_test_split

# Dividir os dados com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    filtered_text.toarray(), y, stratify=y, random_state=42
)

# Treinar o modelo Naive Bayes
nb.fit(X_train, y_train)

# Imprimir a acurácia no conjunto de teste
print(nb.score(X_test, y_test))
