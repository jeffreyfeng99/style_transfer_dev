import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paths to images
style_path = './wave.jpg'
img_path = './axolotl.png'


# Display initial images
style_img = Image.open(style_path).resize((256,256))
content_img = Image.open(img_path).resize((256,256))
plt.subplot(1,2,1)
plt.imshow(style_img)
plt.subplot(1,2,2)
plt.imshow(content_img)
plt.show()


# Load images and preprocess
img_transforms = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

def img_loader(img_path):
    image = Image.open(img_path).convert('RGB')
    image = img_transforms(image)
    image = image.unsqueeze(0)
    return image.to(device)

style_image = img_loader(style_path) 
content_image = img_loader(img_path) 


# Load pretrained model
vgg_model = models.vgg19(pretrained=True).features.to(device).eval()
vgg19_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1,1,1)
vgg19_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1,1,1)


# Redefine the model with incorporated modules for selected layers
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        self.mean = mean
        self.std = std

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
#content_layers = ['conv_14']
#style_layers = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

content_losses = []
style_losses = []

normalization = Normalization(vgg19_mean, vgg19_std).to(device)
model = nn.Sequential(normalization)

i = 0
for layer in vgg_model.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = 'conv_{}'.format(i)
    elif isinstance(layer, nn.ReLU):
        name = 'relu_{}'.format(i)
        layer = nn.ReLU(inplace=False) # Recommended by pytorch
    elif isinstance(layer, nn.MaxPool2d):
        name = 'pool_{}'.format(i)
    elif isinstance(layer, nn.BatchNorm2d):
        name = 'bn_{}'.format(i)
    else:
        raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_image).detach()
        content_loss = ContentLoss(target)
        model.add_module("content_loss_{}".format(i), content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        # add style loss:
        target_feature = model(style_image).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module("style_loss_{}".format(i), style_loss)
        style_losses.append(style_loss)
        
        
# run style transfer
input_image = img_loader(img_path) 
optimizer = optim.LBFGS([input_image.requires_grad_()])  
plt.imshow(np.asarray(input_image.detach().squeeze(0)).transpose(1,2,0))
plt.show()
for i in range(1,200):
    def closure():
        # correct the values of updated input image
        global input_image
        input_image.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_image)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= 1000000
        content_score *= 1

        loss = style_score + content_score
        loss.backward(retain_graph = True)

        if i % 4 == 0:
            print("run {}:".format(i))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_score.item(), content_score.item()))
            print()
        
        return style_score + content_score
    print(i)
    optimizer.step(closure)
    plt.imshow(np.asarray(input_image.detach().squeeze(0)).transpose(1,2,0))
    plt.show()
 
input_image.data.clamp_(0, 1)

plt.imshow(np.asarray(input_image.detach().squeeze(0)).transpose(1,2,0))
plt.show()
image = Image.fromarray(np.asarray(input_image.detach().squeeze(0)).transpose(1,2,0),'RGB')
image.save('./result.jpg')



