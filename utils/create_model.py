# 这里我就先用这种lowB方法实现下, 大佬们好像都是用的装饰器注册,我实在是不会写python
from models.gan import Generator, Discriminator
from models.dcgan import DCGenerator, DCDiscriminator


def create_model(model_name):
    model_name = model_name.lower()
    if model_name == 'gan':
        g_net, d_net = Generator, Discriminator
    elif model_name == 'dcgan':
        g_net, d_net = DCGenerator, DCDiscriminator
    else:
        raise Exception(f'GAN type {model_name} not yet supported.')

    return g_net, d_net
