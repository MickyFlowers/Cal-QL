import timm

model = timm.create_model('resnet50', pretrained=True, framework='jax')
