import tensorflow.keras as tk
import matplotlib.pyplot as plt

# 1. mnsit 데이터 살펴보기
mnist = tk.datasets.mnist


# 이미지, 정답 // 이미지 정답
((train_images, train_labels), (test_images, test_labels)) = mnist.load_data()

# print(train_images)


# 2. 데이터의 shape을 출력
print(train_images.shape) # 60000, 28, 28 (Batch, H, W)
print(train_labels.shape) # 60000 (정답 수)

print(test_images.shape) # 10000, 28, 28
print(test_images.shape) # 10000


# 3. (28, 28) 형태의 이미지를 plt 출력

# train_images[6500].shape
plt.figure() # 640 * 480
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
plt.show()