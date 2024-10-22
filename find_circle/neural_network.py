import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import cv2


# 1. Генерация синтетических данных
class CircleDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = []
        self.labels = []
        self.generate_data()

    def generate_data(self):
        for _ in range(self.num_samples):
            # Решаем случайно, будет ли на изображении круг
            has_circle = np.random.choice([0, 1])
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

            if has_circle:
                # Генерируем случайные параметры круга
                radius = np.random.randint(5, self.img_size // 2)
                center = (
                    np.random.randint(radius, self.img_size - radius),
                    np.random.randint(radius, self.img_size - radius)
                )
                color = np.random.randint(100, 255)
                thickness = -1  # Залитый круг
                cv2.circle(img, center, radius, color, thickness)
            else:
                # Генерируем случайные линии
                for _ in range(np.random.randint(1, 5)):
                    start_point = (
                        np.random.randint(0, self.img_size),
                        np.random.randint(0, self.img_size)
                    )
                    end_point = (
                        np.random.randint(0, self.img_size),
                        np.random.randint(0, self.img_size)
                    )
                    color = np.random.randint(100, 255)
                    thickness = np.random.randint(1, 3)
                    cv2.line(img, start_point, end_point, color, thickness)

            self.data.append(img)
            self.labels.append(has_circle)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        # Нормализуем и добавляем канал
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return torch.tensor(img), torch.tensor(label, dtype=torch.long)


# Создание обучающего и тестового наборов данных
train_dataset = CircleDataset(num_samples=800)
test_dataset = CircleDataset(num_samples=200)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 2. Определение архитектуры модели
class CircleNet(nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # Вход: 1 канал, выход: 16 каналов
            nn.ReLU(),
            nn.MaxPool2d(2),  # Снижение размера изображения вдвое
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 класса: круг или нет
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc_layers(x)
        return x


# Инициализация модели, функции потерь и оптимизатора
model = CircleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Обучение модели
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Поколение {epoch + 1}/{num_epochs}, Потери: {running_loss / len(train_loader):.4f}")

# 4. Оценка модели
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Точность на тестовых данных: {100 * correct / total:.2f}%")


# Функция для отображения изображений
def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i in range(num_images):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"Класс: {label}")
        axes[i].axis('off')
    plt.show()


# Отображение изображений из обучающего набора
show_images(train_dataset)
