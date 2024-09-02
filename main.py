import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL.Image as Image
import os
import glob
import matplotlib.pyplot as plt
import U_Segnet
from torchvision import transforms

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JueyuanziDataset(Dataset):
    def __init__(self, img_path='img', mask_path='mask', transform=None, target_transform=None):
        super().__init__()
        self.img_path = glob.glob(os.path.join(img_path, '*.png'))
        self.mask_path = glob.glob(os.path.join(mask_path, '*.png'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_path)  # 数据集长度

    def __getitem__(self, index):
        x_path = self.img_path[index]
        y_path = self.mask_path[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y


def train_model(model, criterion, optimizer, dataload, num_epochs=100):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.6f " % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
            # with open('U_Segnet_log.txt','a') as f:
            #     f.write("%d/%d,train_loss:%0.6f \n" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.6f" % (epoch, epoch_loss))
        # with open('U_Segnet_sum_loss.txt', 'a') as f:
        #     f.write("epoch %d loss:%0.6f \n" % (epoch, epoch_loss))
        # torch.save(model.state_dict(), 'U_Segnet_weights_%d.pth' % epoch)
    return model


# 网络参数
batch_size = 1
epoch_num = 100

# 加载数据
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
y_transforms = transforms.ToTensor()

dataset = JueyuanziDataset(transform=x_transforms, target_transform=y_transforms)

# 划分数据集
train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])
train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

usegnet = U_Segnet.U_SegNet(3, 1).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(usegnet.parameters(), lr=0.5)

for epoch in range(epoch_num):
    # 创建列表记录数据
    epoch_loss = 0

    for x, y in train_iter:
        inputs = x.to(device)
        labels = y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = usegnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    with open('U_Segnet_sum_loss.txt', 'a') as f:
        f.write("epoch %d loss:%0.6f \n" % (epoch, epoch_loss))

    if epoch % 10 == 0:
        print(f'第{epoch}轮，loss:{epoch_loss / len(train_iter)}')

torch.save(usegnet.state_dict(), 'checkpoints/usegnet_weights.pth')

# 测试网络预测并展示
# usegnet.load_state_dict(torch.load('checkpoints/usegnet_weights.pth', weights_only=True))
#
# show_tensor_pic = transforms.ToPILImage()
# img, mask = next(iter(test_iter))
# plt.figure('image')
# plt.imshow(img[0].permute(1, 2, 0))
# plt.show()
# plt.imshow(mask[0].permute(1, 2, 0))
# plt.show()
# predict_pic = usegnet(img[0].unsqueeze(0).to(device))
# plt.imshow(predict_pic.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
# plt.show()
