import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import model.resnet as resnet


class background_resnet(nn.Module):
    def __init__(self, embedding_size, num_classes, backbone='resnet18', mode='t'):
        super(background_resnet, self).__init__()
        self.trainMode = mode
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=False)
        elif backbone == 'resnet18':
            self.pretrained = resnet.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.pretrained = resnet.resnet34(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
            
        self.fc0 = nn.Linear(128, embedding_size[0])

        # task specific layers for task 1
        self.fc1 = nn.Linear(128, embedding_size[1])
        self.bn1 = nn.BatchNorm1d(embedding_size[1])
        self.relu1 = nn.ReLU()
        self.last1 = nn.Linear(embedding_size[1], num_classes)

        # task speicific layers for task 2
        self.fc2 = nn.Linear(128, embedding_size[2])
        self.bn2 = nn.BatchNorm1d(embedding_size[2])
        self.relu2 = nn.ReLU()
        self.last2 = nn.Linear(embedding_size[2], num_classes)

    def forward(self, x):
        # input x: minibatch x 1 x 40 x 40
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        
        out = F.adaptive_avg_pool2d(x,1) # [batch, 128, 1, 1]
        out = torch.squeeze(out) # [batch, n_embed]
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), -1) # (n_batch, n_embed)
        spk_embedding_shared = self.fc0(out)
        # out = F.relu(self.bn0(spk_embedding)) # [batch, n_embed]
        # out = self.last(out)
        outputs = []
        embeddings = []
        embeddings.append(spk_embedding_shared)
        if(self.trainMode=='i' or self.trainMode=='t'):
            out1 = F.adaptive_avg_pool2d(x, 1)  # [batch, 128, 1, 1]
            out1 = torch.squeeze(out1)  # [batch, n_embed]
            # flatten the out so that the fully connected layer can be connected from here
            out1 = out1.view(x.size(0), -1)  # (n_batch, n_embed)
            spk_embedding_1 = self.fc1(out1)
            out1 = F.relu(self.bn1(spk_embedding_1))  # [batch, n_embed]
            out1 = self.last1(out1)
            outputs.append(out1)
            embeddings.append(spk_embedding_1)
        if(self.trainMode=='c' or self.trainMode=='t'):
            out2 = F.adaptive_avg_pool2d(x, 1)  # [batch, 128, 1, 1]
            out2 = torch.squeeze(out2)  # [batch, n_embed]
            # flatten the out so that the fully connected layer can be connected from here
            out2 = out2.view(x.size(0), -1)  # (n_batch, n_embed)
            spk_embedding_2 = self.fc2(out2)
            out2 = F.relu(self.bn2(spk_embedding_2))  # [batch, n_embed]
            out2 = self.last2(out2)
            outputs.append(out2)
            embeddings.append(spk_embedding_2)
        return embeddings, outputs