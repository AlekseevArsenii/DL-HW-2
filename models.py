
import torch
import torch.nn as nn
import pretrainedmodels
import timm


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=4, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class se_resnext50_32x4d(nn.Module):
    def __init__(self):
        super().__init__()

        # Загружаем модель без классификатора
        base_model = timm.create_model("seresnext50_32x4d", pretrained=True, num_classes=0, global_pool='')

        # Оставляем всё, кроме последнего global pooling (равноценно children()[:-2])
        self.model_ft = nn.Sequential(*list(base_model.children())[:-2])

        # Своя агрегация
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # BatchNorm и BinaryHead, как в оригинале
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(4, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)
        return output
