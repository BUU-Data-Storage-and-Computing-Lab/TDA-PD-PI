import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import copy
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import seaborn as sns

# 设置随机种子以保证可重复性
torch.manual_seed(42)

# 数据预处理，图像大小调整和转换为张量
data_transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}

# 数据集路径
data_dir_dataset1 = '/home/yanghan/桌面/finalcode/2024-DS/2024-intel'
data_dir_dataset2 = '/home/yanghan/桌面/finalcode/2024-DS/PI-intel'

# 创建数据集实例
image_datasets1 = {x: datasets.ImageFolder(root=f'{data_dir_dataset1}/{x}', transform=data_transform[x])
                   for x in ['train', 'test', 'val']}
image_datasets2 = {x: datasets.ImageFolder(root=f'{data_dir_dataset2}/{x}', transform=data_transform[x])
                   for x in ['train', 'test', 'val']}

# 创建数据加载器
train_dataloader1 = DataLoader(image_datasets1['train'], batch_size=32, shuffle=True, num_workers=4)
test_dataloader1 = DataLoader(image_datasets1['test'], batch_size=32, shuffle=False, num_workers=4)
val_dataloader1 = DataLoader(image_datasets1['val'], batch_size=32, shuffle=False, num_workers=4)

train_dataloader2 = DataLoader(image_datasets2['train'], batch_size=32, shuffle=True, num_workers=4)
test_dataloader2 = DataLoader(image_datasets2['test'], batch_size=32, shuffle=False, num_workers=4)
val_dataloader2 = DataLoader(image_datasets2['val'], batch_size=32, shuffle=False, num_workers=4)

dataloaders1 = {'train': train_dataloader1, 'test': test_dataloader1, 'val': val_dataloader1}
dataloaders2 = {'train': train_dataloader2, 'test': test_dataloader2, 'val': val_dataloader2}

# 使用GPU如果可用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载预训练的 VGG16 模型
model1 = models.vgg16(pretrained=True)
model2 = models.vgg16(pretrained=True)
# 冻结模型1的参数
for param in model1.parameters():
    param.requires_grad = False
# 截取到所有卷积层和池化层，包括最后一个最大池化层
model1 = nn.Sequential(*list(model1.features.children()))
model2 = nn.Sequential(*list(model2.features.children()))

# 将模型移动到GPU上
model1 = model1.to(device)
model2 = model2.to(device)

# 定义SE注意力机制模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _,  _= x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y

# 获取特征图的辅助函数
def get_feature_maps(model, dataloader, device):
    feature_maps = []
    labels_list = []

    model.eval()  # 将模型设置为评估模式

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Extracting Features'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            feature_maps.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return feature_maps, labels_list

# 获取训练集测试集验证集的特征图（数据集1）
train_feature_maps1, train_labels1 = get_feature_maps(model1, dataloaders1['train'], device)
test_feature_maps1, test_labels1 = get_feature_maps(model1, dataloaders1['test'], device)
val_feature_maps1, val_labels1 = get_feature_maps(model1, dataloaders1['val'], device)
# 获取训练集测试集验证集的特征图（数据集2）
train_feature_maps2, train_labels2 = get_feature_maps(model2, dataloaders2['train'], device)
test_feature_maps2, test_labels2 = get_feature_maps(model2, dataloaders2['test'], device)
val_feature_maps2, val_labels2 = get_feature_maps(model2, dataloaders2['val'], device)
# 删除样本数不匹配的特征图(训练集)
indices_to_keep = [i for i, (fm1, fm2) in enumerate(zip(train_feature_maps1, train_feature_maps2)) if
                   fm1.shape[0] == fm2.shape[0] == 32]
# 删除样本数不匹配的特征图（测试集）
indices_to_keep_test = [i for i, (fm1, fm2) in enumerate(zip(test_feature_maps1, test_feature_maps2)) if
                        fm1.shape[0] == fm2.shape[0] == 32]
# 删除样本数不匹配的特征图（验证集）
indices_to_keep_val = [i for i, (fm1, fm2) in enumerate(zip(val_feature_maps1, val_feature_maps2)) if
                       fm1.shape[0] == fm2.shape[0] == 32]
# 仅保留匹配的特征图和标签（训练集）
train_feature_maps1 = [train_feature_maps1[i] for i in indices_to_keep]
train_feature_maps2 = [train_feature_maps2[i] for i in indices_to_keep]
train_labels1 = [train_labels1[i] for i in indices_to_keep]
# 仅保留匹配的特征图和标签（测试集）
test_feature_maps1 = [test_feature_maps1[i] for i in indices_to_keep_test]
test_feature_maps2 = [test_feature_maps2[i] for i in indices_to_keep_test]
test_labels1 = [test_labels1[i] for i in indices_to_keep_test]
# 仅保留匹配的特征图和标签（验证集）
val_feature_maps1 = [val_feature_maps1[i] for i in indices_to_keep_val]
val_feature_maps2 = [val_feature_maps2[i] for i in indices_to_keep_val]
val_labels1 = [val_labels1[i] for i in indices_to_keep_val]

# 将两个特征图按通道拼接
train_feature_maps = torch.cat(
    (torch.tensor(np.concatenate(train_feature_maps1, axis=0)), torch.tensor(np.concatenate(train_feature_maps2, axis=0))),
    dim=1)
test_feature_maps = torch.cat(
    (torch.tensor(np.concatenate(test_feature_maps1, axis=0)), torch.tensor(np.concatenate(test_feature_maps2, axis=0))),
    dim=1)
val_feature_maps = torch.cat(
    (torch.tensor(np.concatenate(val_feature_maps1, axis=0)), torch.tensor(np.concatenate(val_feature_maps2, axis=0))),
    dim=1)

# 创建新的 SE 注意力机制模块
se_block = SEBlock(channels=1024, reduction_ratio=16)

# 应用 SE 注意力机制到特征图上
train_feature_maps_attention = se_block(train_feature_maps)
test_feature_maps_attention = se_block(test_feature_maps)
val_feature_maps_attention = se_block(val_feature_maps)

# 将特征图展平
train_feature_maps_flatten = train_feature_maps_attention.view(-1, 50176)
test_feature_maps_flatten = test_feature_maps_attention.view(-1, 50176)
val_feature_maps_flatten = val_feature_maps_attention.view(-1, 50176)

# 创建新的 DataLoader 用于训练
lb = np.concatenate(train_labels1, axis=0)
combined_train_dataset = torch.utils.data.TensorDataset(train_feature_maps_flatten.detach(),
                                                         torch.tensor(np.concatenate(train_labels1, axis=0)).long().detach())
combined_train_dataloader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True, num_workers=0)

# 创建新的 DataLoader 用于测试
combined_test_dataset = torch.utils.data.TensorDataset(test_feature_maps_flatten,
                                                        torch.tensor(np.concatenate(test_labels1, axis=0)).long())
combined_test_dataloader = DataLoader(combined_test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 创建新的 DataLoader 用于验证
combined_val_dataset = torch.utils.data.TensorDataset(val_feature_maps_flatten,
                                                       torch.tensor(np.concatenate(val_labels1, axis=0)).long())
combined_val_dataloader = DataLoader(combined_val_dataset, batch_size=32, shuffle=False, num_workers=0)

combined_dataloaders = {'train': combined_train_dataloader, 'test': combined_test_dataloader,
                        'val': combined_val_dataloader}

# 定义全连接层模型
class FCModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建新的全连接层模型实例
fc_model = FCModel(input_size=50176, num_classes=len(image_datasets1['train'].classes)).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，多类别分类
# 定义优化器
optimizer = optim.SGD(fc_model.parameters(), lr=0.1)
# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=10, gamma=0.05)

# 训练新的模型
num_epochs = 50
best_val_acc_fc = 0.0
best_test_acc = 0.0
best_epoch_fc = 0
best_model_weights_fc = None
train_losses = []
val_losses = []
# 用于存储预测错误的图像文件名
wrong_predictions = []
for epoch in range(num_epochs):
    # 在每个epoch之前更新学习率
    scheduler.step()
    for phase in ['train', 'val']:
        if phase == 'train':
            fc_model.train()
            dataloader = combined_dataloaders['train']
            dataset_size = len(dataloader.dataset)
        else:
            fc_model.eval()
            dataloader = combined_dataloaders['val']
            dataset_size = len(dataloader.dataset)

        running_loss = 0.0
        corrects = 0

        for inputs, labels in tqdm(dataloader, desc=f'{phase} Epoch {epoch}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = fc_model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = corrects.to(torch.float) / dataset_size

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 存储训练和验证损失
        if phase == 'train':
            train_losses.append(epoch_loss)
        else:
            val_losses.append(epoch_loss)

        # 在验证集上寻找最佳模型
        if phase == 'val' and epoch_acc > best_val_acc_fc:
            best_val_acc_fc = epoch_acc
            best_epoch_fc = epoch
            best_model_weights_fc = copy.deepcopy(fc_model.state_dict())

# 打印准确率及对应的 epoch
print(f'Best val Accuracy (with attention): {best_val_acc_fc:.4f} at Epoch {best_epoch_fc}')
# 保存最佳模型权重
torch.save(best_model_weights_fc,
           f'best_model_weights_epoch_{best_epoch_fc}_val_acc_{best_val_acc_fc:.4f}.pth')
# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss (FC)')
plt.plot(val_losses, label='Validation Loss (FC)')
plt.title('Training and Validation Loss (FC)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')  # 使用对数刻度
plt.show()

# 在测试集上测试
fc_model.load_state_dict(best_model_weights_fc)
fc_model.eval()
corrects_fc = 0
all_preds_fc = []
all_labels_fc = []
# 修改这里，使用正确的预测值和真实标签
label_map = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

with torch.no_grad():
    for inputs, labels in tqdm(combined_dataloaders['test'], desc='Testing (with attention)'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = fc_model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects_fc += torch.sum(preds == labels.data)
        all_preds_fc.extend(preds.cpu().numpy())
        all_labels_fc.extend(labels.cpu().numpy())

all_preds_fc = np.array(all_preds_fc)
test_acc_fc = corrects_fc.double() / len(combined_dataloaders['test'].dataset)
precision_fc = precision_score(all_labels_fc, all_preds_fc, average='weighted', zero_division=0)
recall_fc = recall_score(all_labels_fc, all_preds_fc, average='weighted', zero_division=0)
f1_fc = f1_score(all_labels_fc, all_preds_fc, average='weighted', zero_division=0)
print(f'Test Accuracy: {test_acc_fc:.4f}')
print(f'Precision: {precision_fc:.4f}')
print(f'Recall: {recall_fc:.4f}')
print(f'F1 Score: {f1_fc:.4f}')
print("Training and Testing complete!")
# 获取真实标签和预测标签
y_true = np.array(all_labels_fc)
y_pred = np.array(all_preds_fc)
# 计算混淆矩阵
conf_mat = confusion_matrix(y_true, y_pred)

# 绘制混淆矩阵热力图
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')

# 添加轴标签和标题
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# 显示图表
plt.show()

# 创建颜色条
cbar = heatmap.collections[0].colorbar
cbar.set_label('Count')  # 设置颜色条标签

# 输出预测错误的图像文件名
print("\nWrong Predictions:")
# 修改这里，添加错误预测图像文件名到列表
num_wrong_predictions = 0
for inputs, labels in combined_dataloaders['test']:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = fc_model(inputs)
    _, preds = torch.max(outputs, 1)
    wrong_predictions.extend([image_datasets1['test'].imgs[i][0] for i, p in enumerate(preds) if p != labels[i]])
    num_wrong_predictions += torch.sum(preds != labels.data)
wrong_predictions = list(set(wrong_predictions))  # 去除重复的错误预测文件名
print("Number of wrong predictions:", num_wrong_predictions)
print("First few wrong predictions:", wrong_predictions[:10])  # 输出前10个错误预测的图像文件名

