"""PlantXViT re-implementation (Thakur et al., Ecological Informatics 77, 102245, 2023).

Serial design described by the authors: a convolutional stem (first VGG16 blocks + an Inception-style
block) extracts local features, which are split into small patches, linearly embedded, and processed by a
lightweight vision-transformer encoder; the class is predicted from the pooled token sequence.
Hyper-parameters below follow the published description as closely as it allows; check them against the
original paper before reporting (they are exposed as constructor arguments).
"""
import torch
import torch.nn as nn


def conv_bn(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_branch=32):
        super().__init__()
        self.b1 = conv_bn(c_in, c_branch, 1)
        self.b3 = nn.Sequential(conv_bn(c_in, c_branch, 1), conv_bn(c_branch, c_branch, 3))
        self.b5 = nn.Sequential(conv_bn(c_in, c_branch, 1), conv_bn(c_branch, c_branch, 5))
        self.bp = nn.Sequential(nn.MaxPool2d(3, 1, 1), conv_bn(c_in, c_branch, 1))
        self.out_channels = 4 * c_branch

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], 1)


class PlantXViT(nn.Module):
    def __init__(self, num_classes=38, img_size=224, patch=5, dim=64, depth=4, heads=4, mlp_ratio=2,
                 dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(  # VGG16-style blocks 1-2
            conv_bn(3, 64), conv_bn(64, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2))
        self.incep = InceptionBlock(128, 32)
        fmap = img_size // 4                                      # 56 for 224 input
        fmap_p = (fmap // patch) * patch
        self.crop = fmap_p
        self.embed = nn.Conv2d(self.incep.out_channels, dim, patch, patch)  # patch embedding
        n_tok = (fmap_p // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * mlp_ratio, dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.dropout, self.fc = nn.Dropout(dropout), nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.incep(self.stem(x))[:, :, :self.crop, :self.crop]
        t = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        t = self.norm(self.encoder(t)).mean(1)
        return self.fc(self.dropout(t))
