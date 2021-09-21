from src.KNN import KNN

header = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
dataset_path = '~/Downloads/iris.data'
knn = KNN(dataset_path, header)
knn.load_data_set()
knn.calibration(10)
novos_exemplos = [[1.6,0.5,5.0,3.6],
                  [4.2,1.2,5.8,2.7],
                  [5.2,2.4,7.0,3.2]]
print(knn.predict_data(novos_exemplos))
