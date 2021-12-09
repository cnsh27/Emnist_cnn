from emnist import extract_training_samples, extract_test_samples
import matplotlib.pyplot as plt
# import numpy as np

# from keras.utils import np_utils

# byclass를 사용
xTrain, yTrain = extract_training_samples('byclass')
xTest, yTest = extract_test_samples('byclass')

# # 1~255값으로 표현되는 데이터를 255로 나눠줌으로써 머신러닝에 최적화된 0~1의 데이터로 변환
# xTrain = np.array(xTrain) / 255.0
# yTrain = np.array(yTrain)
# xTest = np.array(xTest) / 255.0
# yTest = np.array(yTest)

# # 데이터를 획일화, 모두 28, 28 로 변환
# xTrain = xTrain.reshape(xTrain.shape[0], 28, 28)
# xTest = xTest.reshape(xTest.shape[0], 28, 28)

# 학습 데이터 중 표본 보기
plt.imshow(xTrain[0])
plt.show()
print(repr(xTrain[0]))

# # 행렬을 바꾼다. 
# for i in range(len(xTrain)):
#     xTrain[i] = np.transpose(xTrain[i])
# for i in range(len(xTest)):
#     xTest[i] = np.transpose(xTest[i])

# # (28, 28) => (784, 1) Flatten과 같은 역할
# xTrain.reshape(xTrain.shape[0], 784, 1) 
# xTest.reshape(xTest.shape[0], 784, 1) 

# # 전처리 완료

# # 데이터를 1차원 np 배열로 변환
# def resh(imgs):
#     data = []
#     for img in imgs:
#         data.append(img.reshape(-1))
#     return np.asarray(data)

# # 데이터 형식의 변환
# train_images = xTrain.astype('float32')
# test_images = xTest.astype('float32')

# # Flatten의 데이터 변환
# train_images = resh(train_images)
# test_images = resh(test_images)

# train_labels = np_utils.to_categorical(yTrain, 62)
# test_labels = np_utils.to_categorical(yTest, 62)


# print("yTrain ", yTrain[0])
# print("\n\n\n\n\n\n")
# print("label ", train_labels[0])