with open('iris.csv', 'r', encoding='utf-8') as f:
    iris = f.readlines()
iris = [line.strip().split(',') for line in iris[1:]]
    
print(iris)