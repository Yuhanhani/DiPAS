import torch
import torch.nn as nn

# follow exactly the same manner as original paper, stride=2 and no padding
# if want output size same as input, make stride=1, padding=1 and choose input number which is divisible by 16
# remeber to cite https://becominghuman.ai/implementing-unet-in-pytorch-8c7e05a121b4


# double 3x3 convolution
def dual_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


# crop the image(tensor) to equal size
# as shown in architecture image , half left side image is concated with right side image
def crop_tensor(target_tensor, tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Left side (contracting path)
        self.dwn_conv1 = dual_conv(3, 64)
        self.dwn_conv2 = dual_conv(64, 128)
        self.dwn_conv3 = dual_conv(128, 256)
        self.dwn_conv4 = dual_conv(256, 512)
        self.dwn_conv5 = dual_conv(512, 1024)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Right side  (expnsion path)
        # transpose convolution is used showna as green arrow in architecture image
        self.trans1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = dual_conv(1024, 512)
        self.trans2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = dual_conv(512, 256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = dual_conv(256, 128)
        self.trans4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = dual_conv(128, 64)

        # output layer
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, image):
        # forward pass for Left side
        x1 = self.dwn_conv1(image)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)


        # forward pass for Right side
        x = self.trans1(x9)
        y = crop_tensor(x, x7)
        x = self.up_conv1(torch.cat([x, y], 1))

        x = self.trans2(x)
        y = crop_tensor(x, x5)
        x = self.up_conv2(torch.cat([x, y], 1))

        x = self.trans3(x)
        y = crop_tensor(x, x3)
        x = self.up_conv3(torch.cat([x, y], 1))

        x = self.trans4(x)
        y = crop_tensor(x, x1)
        x = self.up_conv4(torch.cat([x, y], 1))

        x = self.out(x)

        return x


if __name__ == '__main__':
    image = torch.rand((1, 3, 572, 572))  # if change input size, needs to recalculate all sizes otherwise may lead to decimal numbers hence unmatch
    model = Unet()
    x = model(image)
    print(x.size())

# warning means needs to change to a stronger cpu/gpu in order to use nnpack
# can use export USE_NNPACK=0 command to prepend it first
# see website https://discuss.pytorch.org/t/bug-w-nnpack-cpp-80-could-not-initialize-nnpack-reason-unsupported-hardware/107518/5