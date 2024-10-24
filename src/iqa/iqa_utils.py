import os
import numpy as np
import torch.nn as nn
#增加

# from aige_EfficientNet.build_model import Net as single_model
# from aige_EfficientNet.attention_torch.attention.ShuffleAttention import ShuffleAttention

class Efficientnet_Sngle(nn.Module):
    def __init__(self,train_classes=10):
        super(Efficientnet_Sngle, self).__init__()
        self.model_one = single_model(model_name='efficientnet-b0', classifications=1000, pretrained=True)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(0.4)
        self.channel = 1280
        #2560-->3584-->5120
        self._fc = nn.Linear(self.channel,train_classes)
        self._fc_tanh = nn.Tanh()
        self.se = ShuffleAttention(channel=self.channel,G=8)

    def forward(self,input_one):
        feature_one = self.model_one.extract_features(input_one)
        output = self.se(feature_one)
        output = self._avg_pooling(output)
        output = output.flatten(start_dim=1)
      
        output = self._dropout(output)
        output = self._fc(output)

        #多标签输出
        return output

def get_network(backbone, cls):
    if backbone == 'resnet18':
        from src.iqa.models.resnet import resnet18
        model_ft = resnet18()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, cls)
    elif backbone == 'efficientnet-b0':
        from aige_EfficientNet.build_model import Net as single_model
        from aige_EfficientNet.attention_torch.attention.ShuffleAttention import ShuffleAttention
        model_ft = Efficientnet_Sngle(train_classes=cls)
    else:
        raise NotImplementedError(f"{backbone} has not been implemented...")
    
    net = model_ft
    return net
    
    # elif model_name == 'shufflenet':
    #     model = models.shufflenet_v2_x0_5(pretrained=pretrained)
    #     num_ftrs = model.fc.in_features
    #     model.fc = nn.Linear(num_ftrs,classifications)
    # elif model_name == 'efficientnet-b0':
    #     model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=classifications)
    # elif model_name == 'efficientnet-b4':
    #     model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=classifications)  
    # elif model_name == 'efficientnet-b7':
    #     model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=classifications)
    
    
#原图    
# import os
# import numpy as np
# import torch.nn as nn


# def get_network(backbone, cls):
#     if backbone == 'resnet18':
#         from api.aigc_iqa.models.resnet import resnet18
#         model_ft = resnet18()
#         num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, cls)
#     else:
#         raise NotImplementedError('{} has not been implemented...'.format(backbone))
#     net = model_ft

#     return net