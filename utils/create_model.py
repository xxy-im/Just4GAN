# 这里我就先用这种lowB方法实现下, 大佬们好像都是用的装饰器注册,我实在是不会写python
def create_model(model_name):
    model_name = model_name.lower()
    model = __import__('models.'+model_name)
    return model.Generator, model.Discriminator
