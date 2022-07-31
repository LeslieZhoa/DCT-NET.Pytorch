from torch import nn
from torch.nn import functional as F
import torch

# refer to https://github.com/vinesmsuic/White-box-Cartoonization-PyTorch
# VGG architecter, used for the perceptual loss using a pretrained VGG network
VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
class VGG19(torch.nn.Module):
    def __init__(self, in_channels=3,
                 VGGtype="VGG19", 
                 init_weights=None, 
                 batch_norm=False, 
                 num_classes=1000, 
                 feature_mode=False,
                 requires_grad=False):
        super(VGG19, self).__init__()
        self.in_channels = in_channels
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm

        self.features = self.create_conv_layers(VGG_types[VGGtype])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        if init_weights is not None:
            self.load_state_dict(torch.load(init_weights))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if not self.feature_mode:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        elif self.feature_mode == True and self.batch_norm == False:
            module_list = list(self.features.modules())
            #print(module_list[1:27])
            for layer in module_list[1:27]: # conv4_4 Feature maps
                x = layer(x) 
        else:
            raise ValueError('Feature mode does not work with batch norm enabled. Set batch_norm=False and try again.')

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        batch_norm = self.batch_norm

        for x in architecture:
            if type(x) == int: # Number of features
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ]

                if batch_norm == True:
                    # Back at that time Batch Norm was not invented
                    layers += [nn.BatchNorm2d(x),nn.ReLU(),]
                else:
                    layers += [nn.ReLU()]

                in_channels = x #update in_channel

            elif x == "M": # Maxpooling
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)



# refer to https://github.com/vinesmsuic/White-box-Cartoonization-PyTorch
def box_filter( x, r):
    channel =  x.shape[1] # Batch, Channel, H, W
    kernel_size = (2*r+1)
    weight = 1.0/(kernel_size**2)
    box_kernel = weight*torch.ones((channel, 1, kernel_size, kernel_size), dtype=torch.float32, device=x.device)
    output = F.conv2d(x, weight=box_kernel, stride=1, padding=r, groups=channel) #tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')

    return output


def guided_filter(x, y, r, eps=1e-2):
    # Batch, Channel, H, W
    _, _, H, W = x.shape

    N = box_filter(torch.ones((1, 1, H, W), dtype=x.dtype, device=x.device), r)

    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b
    return output

# refer to https://github.com/SystemErrorWang/White-box-Cartoonization
def color_shift(image, mode='uniform'):
    device = image.device
    b1, g1, r1 = torch.split(image, 1, dim=1)
   
    if mode == 'normal':
        b_weight = torch.normal(mean=0.114, std=0.1,size=[1]).to(device)
        g_weight = torch.normal(mean=0.587, std=0.1,size=[1]).to(device)
        r_weight = torch.normal(mean=0.299, std=0.1,size=[1]).to(device)
    elif mode == 'uniform':
        
        b_weight = torch.FloatTensor(1).uniform_(0.014,0.214).to(device)
        g_weight = torch.FloatTensor(1).uniform_(0.487, 0.687).to(device)
        r_weight = torch.FloatTensor(1).uniform_(0.199, 0.399).to(device)
    output1 = (b_weight*b1+g_weight*g1+r_weight*r1)/(b_weight+g_weight+r_weight)
    
    return output1


