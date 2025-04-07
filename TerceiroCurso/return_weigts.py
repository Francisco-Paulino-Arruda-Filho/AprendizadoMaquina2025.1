def return_weights(vocab, vector, vector_index):
    
    zipped = dict(zip(
        vector[vector_index].indices,
        vector[vector_index].dataz
    ))
    return {vocab[i]: zipped[i] for i in zipped.keys() if i in vocab}

print(return_weights(
    vocab = {'a': 0, 'b': 1, 'c': 2}, 
    vector = [[0, 1, 2], [3, 4, 5]], 
    vector_index = 1
))