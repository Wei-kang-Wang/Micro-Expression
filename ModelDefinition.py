from collections import OrderedDict

import torch.nn as nn

class MyModel(nn.Module):

    NUM_CAGETORIES = 7
    
    def __init__(self, resnet_class, output_size=NUM_CAGETORIES):
        super(MyModel, self).__init__()
        pretrained_resnet = resnet_class(pretrained=True)

        self.features = nn.Sequential( OrderedDict([
            ('conv1', pretrained_resnet.conv1),
            ('bn1', pretrained_resnet.bn1),
            ('relu', pretrained_resnet.relu),
            ('maxpool', pretrained_resnet.maxpool),

            ('layer1', pretrained_resnet.layer1),
            ('layer2', pretrained_resnet.layer2),
            ('layer3', pretrained_resnet.layer3),
            ('layer4', pretrained_resnet.layer4),

            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))) # 参数(1, 1)表示希望输出的feature map空间方向上的维度为1×1
        ]) )

        hidden_size_list = []
        cur_size = pretrained_resnet.fc.in_features
        layers = []
        
        for hidden_size in hidden_size_list:
            layers.append( nn.Linear(cur_size, hidden_size, bias=False) )
            layers.append( nn.BatchNorm1d(hidden_size) )
            layers.append( nn.ReLU(True) )
            cur_size = hidden_size
        
        layers.append( nn.Linear(cur_size, output_size) )
        self.classifier = nn.Sequential(*layers)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

    def partial_forward(self, x):
        hidden = self.features[:6](x)
        return hidden
    
    def partial_forward_high(self, x):
        hidden = self.features[:8](x)
        return hidden
    
    def partial_forward_mid_high(self, x):
        mid = self.features[:6](x)
        high = self.features[6:8](mid)
        return mid,high
    
    def forward_with_mid_high(self, x):
        mid = self.features[:6](x)
        high = self.features[6:8](mid)
        output = self.features[8:](high)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return mid,high, output
    
    def forward_with_hidden(self, x):
        hidden = self.features[:6](x)
        output = self.features[6:](hidden)

        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return hidden, output
    
    def forward_with_high(self, x):
        hidden = self.features[:8](x)
        output = self.features[8:](hidden)

        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return hidden, output
    # 参考：https://pytorch.org/docs/master/notes/autograd.html
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            
            
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.seq2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        initialize_weights(self.seq1)
        initialize_weights(self.seq2)


    def forward(self, x):
        x = self.seq1(x)
        x = x.view(x.size(0), -1)
        output = self.seq2(x)
        return output

class Discriminator_mid(nn.Module):

    def __init__(self):
        super(Discriminator_mid, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.seq2 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        initialize_weights(self.seq1)
        initialize_weights(self.seq2)


    def forward(self, x):
        x = self.seq1(x)
        x = x.view(x.size(0), -1)
        output = self.seq2(x)
        return output
    
class Discriminator_high(nn.Module):

    def __init__(self):
        super(Discriminator_high, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.seq2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        initialize_weights(self.seq1)
        initialize_weights(self.seq2)


    def forward(self, x):
        x = self.seq1(x)
        x = x.view(x.size(0), -1)
        output = self.seq2(x)
        return output    
    
def initialize_weights( model ):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)          
            