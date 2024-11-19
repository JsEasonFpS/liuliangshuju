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
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集路径
data_path = "D:\\yanjiusheng\\liuliangshuju\\augment2500\\"

# 测试集路径
test_data_path = "D:\\yanjiusheng\\liuliangshuju\\data\\test\\"  # 在此路径下的文件夹作为测试集

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

# 读取测试集图像并标签化
def read_test_images(test_data_path):
    imgs = []
    labels = []
    for label, folder in enumerate(os.listdir(test_data_path)):
        folder_path = os.path.join(test_data_path, folder)
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

# CNN 模型定义
class CombinedCNN(nn.Module):
    def __init__(self, num_classes=12):  # 假设是12个类别
        super(CombinedCNN, self).__init__()
        
        # Sobel分支
        self.sobel_conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.sobel_conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.sobel_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # LBP分支
        self.lbp_conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.lbp_conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.lbp_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层，输入512维（Sobel和LBP输出拼接后的大小），输出num_classes维
        self.fc1 = nn.Linear(128 * 8 * 8 * 2, 512)  # 输入是128*8*8的输出 * 2（拼接的Sobel和LBP） 
        self.fc2 = nn.Linear(512, num_classes)  # 分类输出层，num_classes是12

    def forward(self, sobel_input, lbp_input):
        # Sobel分支
        sobel_out = F.relu(self.sobel_conv1(sobel_input))
        sobel_out = self.sobel_pool(sobel_out)
        sobel_out = F.relu(self.sobel_conv2(sobel_out))
        sobel_out = self.sobel_pool(sobel_out)
        
        # LBP分支
        lbp_out = F.relu(self.lbp_conv1(lbp_input))
        lbp_out = self.lbp_pool(lbp_out)
        lbp_out = F.relu(self.lbp_conv2(lbp_out))
        lbp_out = self.lbp_pool(lbp_out)
        
        # 融合两者的输出
        combined_out = torch.cat((sobel_out, lbp_out), dim=1)
        
        # 展平拼接后的特征
        combined_out = combined_out.view(-1, 128 * 8 * 8 * 2)  # 展平
        
        # 通过第一个全连接层
        combined_out = F.relu(self.fc1(combined_out))
        
        # 最终分类
        final_output = self.fc2(combined_out)
        
        return final_output

# 定义12个类别的标签
class_names = ['ATTACK', 'DATABASE', 'FTP-CONTROL', 'FTP-DATA', 'FTP-PASV', 'GAMES', 
               'INTERACTIVE', 'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW']


# 计算并打印分类报告
def print_classification_report(all_labels, all_preds):
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=5)
    print(report)

# 读取数据
data, labels = read_images(data_path)
data = data.reshape(-1, 32, 32) / 255.0  # 归一化
sobel_data = np.array([gaussian_sobel(img) for img in data])
lbp_data = np.array([lbp(img) for img in data])

# 划分训练集和验证集
x_train_sobel, x_temp_sobel, y_train, y_temp = train_test_split(sobel_data, labels, test_size=0.2, stratify=labels, random_state=42)
x_train_lbp, x_temp_lbp, _, _ = train_test_split(lbp_data, labels, test_size=0.2, stratify=labels, random_state=42)

x_valid_sobel, x_test_sobel, y_valid, y_test = train_test_split(x_temp_sobel, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
x_valid_lbp, x_test_lbp, _, _ = train_test_split(x_temp_lbp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 创建训练集和验证集数据集和数据加载器
train_dataset = ImageDataset(torch.tensor(x_train_sobel).float(), torch.tensor(y_train).long())
valid_dataset = ImageDataset(torch.tensor(x_valid_sobel).float(), torch.tensor(y_valid).long())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# 创建测试集数据集和数据加载器（使用指定的测试集路径）
test_data, test_labels = read_test_images(test_data_path)
test_data = test_data.reshape(-1, 32, 32) / 255.0  # 归一化
sobel_test_data = np.array([gaussian_sobel(img) for img in test_data])
lbp_test_data = np.array([lbp(img) for img in test_data])

test_dataset = ImageDataset(torch.tensor(sobel_test_data).float(), torch.tensor(test_labels).long())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型、损失函数和优化器
model = CombinedCNN(num_classes=12).to(device)  # 类别数量12
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 训练过程
num_epochs = 100
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

# 混淆矩阵的预测和标签
all_preds = []
all_labels = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for (sobel_inputs, labels), (lbp_inputs, _) in zip(train_loader, train_loader):
        sobel_inputs, labels = sobel_inputs.to(device).view(-1, 1, 32, 32), labels.to(device)
        lbp_inputs = lbp_inputs.to(device).view(-1, 1, 32, 32)

        optimizer.zero_grad()
        # 将Sobel和LBP输入传入模型
        outputs = model(sobel_inputs, lbp_inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    # 验证集评估
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (sobel_inputs, labels), (lbp_inputs, _) in zip(valid_loader, valid_loader):
            sobel_inputs = sobel_inputs.to(device).view(-1, 1, 32, 32)
            lbp_inputs = lbp_inputs.to(device).view(-1, 1, 32, 32)
            labels = labels.to(device)

            outputs = model(sobel_inputs, lbp_inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    valid_loss = total_loss / len(valid_loader)
    valid_accuracy = correct / total
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

# 打印分类报告并计算准确率
report = classification_report(all_labels, all_preds, target_names=class_names, digits=5)
print(report)

# 绘制损失曲线和准确率曲线
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), valid_accuracies, label='Valid Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)

# 计算每个格子的占比
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# 创建一个矩阵，默认填充为空字符串
annot_matrix = [['' for _ in range(len(conf_matrix))] for _ in range(len(conf_matrix))]

# 对于预测正确的格子，替换为百分比格式
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[i])):
        if i == j:  # 只在预测正确的格子（对角线）显示百分比
            annot_matrix[i][j] = f"{conf_matrix_percentage[i, j]:.2f}%"

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_percentage, annot=annot_matrix, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix with Percentages in Correct Predictions')
plt.show()
