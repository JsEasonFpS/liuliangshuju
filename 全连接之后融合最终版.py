import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据集路径
data_path = "D:\\yanjiusheng\\liuliangshuju\\augment2500\\"  # 训练集和验证集数据
test_data_path = "D:\\yanjiusheng\\liuliangshuju\\data\\test\\"  # 给定的测试集路径

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 读取图像并标签化
def read_images(data_path):
    imgs = []
    labels = []
    for label, folder in enumerate(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for img_path in glob.glob(folder_path + '/*.png'):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (32, 32))  # 统一图像大小
                imgs.append(img)
                labels.append(label)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

# Sobel 边缘检测
def gaussian_sobel(image):
    blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return sobel_magnitude.flatten()

# LBP 特征提取
def lbp(image):
    lbp_image = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = image[i, j]
            binary = 0
            for m in range(-1, 2):
                for n in range(-1, 2):
                    if m == 0 and n == 0:
                        continue
                    binary <<= 1
                    binary |= (image[i + m, j + n] >= center)
            lbp_image[i, j] = binary
    return lbp_image.flatten()

# CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 12)  # 12个类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)  # 扁平化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 读取数据
data, labels = read_images(data_path)
data = data.reshape(-1, 32, 32) / 255.0  # 归一化
sobel_data = np.array([gaussian_sobel(img) for img in data])
lbp_data = np.array([lbp(img) for img in data])

# 划分训练集和验证集（4:1的比例）
x_train_sobel, x_valid_sobel, y_train, y_valid = train_test_split(sobel_data, labels, test_size=0.2, stratify=labels, random_state=42)
x_train_lbp, x_valid_lbp, _, _ = train_test_split(lbp_data, labels, test_size=0.2, stratify=labels, random_state=42)

# 创建数据集和数据加载器
train_dataset = ImageDataset(torch.tensor(x_train_sobel).float(), torch.tensor(y_train).long())
valid_dataset = ImageDataset(torch.tensor(x_valid_sobel).float(), torch.tensor(y_valid).long())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # num_workers= pin_memory=
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# 读取测试集
test_data, test_labels = read_images(test_data_path)
test_data = test_data.reshape(-1, 32, 32) / 255.0  # 归一化
sobel_test_data = np.array([gaussian_sobel(img) for img in test_data])
lbp_test_data = np.array([lbp(img) for img in test_data])

# 创建测试集数据集和数据加载器
test_dataset = ImageDataset(torch.tensor(sobel_test_data).float(), torch.tensor(test_labels).long())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型、损失函数和优化器
sobel_model = SimpleCNN().to(device)
lbp_model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(sobel_model.parameters()) + list(lbp_model.parameters()), lr=0.00001)

# 修改后的模型部分：拼接后的输出通过全连接层转换为12类分类
class FusionCNN(nn.Module):
    def __init__(self):
        super(FusionCNN, self).__init__()
        self.fc_fusion = nn.Linear(24, 12)  # 将拼接后的24维特征转换为12类分类

    def forward(self, sobel_outputs, lbp_outputs):
        # 拼接两个输出
        combined_outputs = torch.cat((sobel_outputs, lbp_outputs), dim=1)
        # 通过全连接层进行分类
        final_output = self.fc_fusion(combined_outputs)
        return final_output

# 训练和验证
num_epochs = 100
fusion_model = FusionCNN().to(device)

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in range(num_epochs):
    sobel_model.train()
    lbp_model.train()
    fusion_model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for (sobel_inputs, labels), (lbp_inputs, _) in zip(train_loader, train_loader):
        sobel_inputs, labels = sobel_inputs.to(device).view(-1, 1, 32, 32), labels.to(device)
        lbp_inputs = lbp_inputs.to(device).view(-1, 1, 32, 32)

        optimizer.zero_grad()
        # 分别通过两个模型进行前向传播
        sobel_outputs = sobel_model(sobel_inputs)
        lbp_outputs = lbp_model(lbp_inputs)

        # 融合两个输出并进行分类
        final_output = fusion_model(sobel_outputs, lbp_outputs)

        loss = criterion(final_output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(final_output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    # 验证集评估
    sobel_model.eval()
    lbp_model.eval()
    fusion_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for (sobel_inputs, labels), (lbp_inputs, _) in zip(valid_loader, valid_loader):
            sobel_inputs = sobel_inputs.to(device).view(-1, 1, 32, 32)
            lbp_inputs = lbp_inputs.to(device).view(-1, 1, 32, 32)
            labels = labels.to(device)

            sobel_outputs = sobel_model(sobel_inputs)
            lbp_outputs = lbp_model(lbp_inputs)

            final_output = fusion_model(sobel_outputs, lbp_outputs)

            loss = criterion(final_output, labels)
            total_loss += loss.item()

            _, predicted = torch.max(final_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_loss = total_loss / len(valid_loader)
    valid_accuracy = correct / total
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    print(f'Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}')

# 绘制准确率和损失曲线
plt.figure(figsize=(12, 5))

# 第一幅子图：绘制准确率曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# 第二幅子图：绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 显示所有子图
plt.tight_layout()
plt.show()


# 最终测试集评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 最终测试集评估
fusion_model.eval()
all_predicted = []  # 用于保存所有预测结果
all_labels = []     # 用于保存所有真实标签

with torch.no_grad():
    for (sobel_inputs, labels), (lbp_inputs, _) in zip(test_loader, test_loader):
        sobel_inputs = sobel_inputs.to(device).view(-1, 1, 32, 32)
        lbp_inputs = lbp_inputs.to(device).view(-1, 1, 32, 32)
        labels = labels.to(device)

        sobel_outputs = sobel_model(sobel_inputs)
        lbp_outputs = lbp_model(lbp_inputs)

        final_output = fusion_model(sobel_outputs, lbp_outputs)

        _, predicted = torch.max(final_output.data, 1)

        all_predicted.extend(predicted.cpu().numpy())  # 保存预测结果
        all_labels.extend(labels.cpu().numpy())         # 保存真实标签

# 混淆矩阵
cm = confusion_matrix(all_labels, all_predicted)  # 使用收集的所有标签和预测

# 标签名称
labels = ['ATTACK', 'DATABASE', 'FTP-CO0TROL', 'FTP-DATA', 'FTP-PASV', 'GAMES', 'I0TERACTIVE', 'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW']

# 计算每个类别的准确率并显示
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比

# 绘制混淆矩阵热力图，仅显示对角线上的预测正确百分比
plt.figure(figsize=(10, 7))
plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(12)
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')

# 标注每个格子，显示百分比
for i in range(12):
    for j in range(12):
        if i == j:
            plt.text(j, i, f'{cm_percent[i, j]:.2f}%', ha="center", va="center", color="black")

plt.show()

# 计算评估指标
accuracy = accuracy_score(all_labels, all_predicted)
precision = precision_score(all_labels, all_predicted, average='weighted')
recall = recall_score(all_labels, all_predicted, average='weighted')
f1 = f1_score(all_labels, all_predicted, average='weighted')

# 打印评估指标，保留五位小数
print(f'Accuracy: {accuracy:.5f}')
print(f'Precision: {precision:.5f}')
print(f'Recall: {recall:.5f}')
print(f'F1 Score: {f1:.5f}')
print("\nClassification Report:")
print(classification_report(all_labels, all_predicted, target_names=labels))
