from models.vision_transformer import vit_base

vit = vit_base()
for name, param in vit.named_parameters():
    print(name, param.shape)