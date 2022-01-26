#!/usr/bin/env python
# coding: utf-8

# In[3]:


conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch


# In[4]:


from torchvision import models


# In[5]:


dir(models)


# In[6]:


alexnet = models.AlexNet()


# In[7]:


resnet = models.resnet101(pretrained=True)


# In[8]:


from torchvision import transforms
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])


# In[9]:


from PIL import Image
img = Image.open("../data/p1ch2/bobby.jpg")


# In[10]:


python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow


# In[11]:


from PIL import Image


# In[12]:


img = Image.open("../data/p1ch2/bobby.jpg")


# In[13]:


img


# In[14]:


dir(Image)


# In[15]:


img = Image.open("Desktop/WoW moments/3000.png")


# In[16]:


img


# In[17]:


img_t = preprocess(img)


# In[18]:


from PIL import Image
img = Image.open("bobby.jpg")


# In[19]:


img


# In[20]:


img.show()


# In[21]:


img_t = preprocess(img)


# In[22]:


test_image = Image.open(img).convert('RGB')


# In[23]:


img = Image.open("Desktop/WoW moments/bobby.jpg")


# In[24]:


img


# In[25]:


img_t = preprocess(img)


# In[26]:


import torch
batch_t = torch.unsqueeze(img_t, 0)


# In[27]:


resnet.eval()


# In[28]:


out = resnet(batch_t)
out


# In[29]:


with open('imagenet_classes.txt') as f:
    labels = f.readlines()


# In[30]:


with open('desktop/ML/imagenet_classes.txt') as f:
     labels = f.readlines()


# In[31]:


_, index = torch.max(out, 1)


# In[32]:


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[33]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[34]:


img2 = Image.open("Desktop/ML/cat.jpg")
img3 = Image.open("Desktop/ML/eagle.jpg")
img4 = Image.open("Desktop/ML/horse.jpg")
img5 = Image.open("Desktop/ML/gator.jpg")


# In[35]:


img2


# In[36]:


img_t2 = preprocess(img2)


# In[37]:


batch_t = torch.unsqueeze(img_t2, 0)


# In[38]:


_, index = torch.max(out, 1)


# In[39]:


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[40]:


percentage2 = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage2[index[0]].item()


# In[41]:


out = resnet(batch_t)


# In[42]:


percentage2 = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage2[index[0]].item()


# In[43]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[44]:


img_t3 = preprocess(img3)


# In[45]:


batch_t = torch.unsqueeze(img_t3, 0)


# In[46]:


out = resnet(batch_t)


# In[47]:


percentage3 = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage2[index[0]].item()


# In[154]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage3[idx].item()) for idx in indices[0][:5]]


# In[49]:


img_t4 = preprocess(img4)


# In[50]:


batch_t = torch.unsqueeze(img_t4, 0)


# In[51]:


out = resnet(batch_t)


# In[52]:


percentage3 = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage2[index[0]].item()


# In[53]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[54]:


img_t5 = preprocess(img5)


# In[55]:


batch_t = torch.unsqueeze(img_t5, 0)


# In[56]:


out = resnet(batch_t)


# In[57]:


percentage3 = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage2[index[0]].item()


# In[58]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[59]:


netG = ResNetGenerator()


# In[60]:


ResNetGenerator()


# In[61]:


import torch
import torch.nn as nn

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)


# In[62]:


netG = ResNetGenerator()


# In[64]:


model_path = 'desktop/ML/data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)


# In[65]:


netG.eval()


# In[66]:


from PIL import Image
from torchvision import transforms


# In[67]:


preprocess = transforms.Compose([transforms.Resize(256),
transforms.ToTensor()])


# In[68]:


img = Image.open("desktop/ML/data/p1ch2/horse.jpg")
img


# In[69]:


img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)


# In[70]:


batch_out = netG(batch_t)


# In[71]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)


# In[72]:


out_img


# In[74]:


img = Image.open("desktop/ML/horse.jpg")
img2 = Image.open("desktop/ML/horse2.jpg")
img3 = Image.open("desktop/ML/horse3.jpg")
img4 = Image.open("desktop/ML/horse4.jpg")
img5 = Image.open("desktop/ML/horse5.jpg")


# In[75]:


img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)


# In[76]:


batch_out = netG(batch_t)


# In[77]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img


# In[80]:


img_t = preprocess(img2)
batch_t = torch.unsqueeze(img_t, 0)


# In[81]:


batch_out = netG(batch_t)


# In[82]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img


# In[83]:


img_t = preprocess(img3)
batch_t = torch.unsqueeze(img_t, 0)


# In[84]:


batch_out = netG(batch_t)


# In[85]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img


# In[86]:


img_t = preprocess(img4)
batch_t = torch.unsqueeze(img_t, 0)


# In[87]:


batch_out = netG(batch_t)


# In[88]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img


# In[89]:


img_t = preprocess(img5)
batch_t = torch.unsqueeze(img_t, 0)


# In[90]:


batch_out = netG(batch_t)


# In[91]:


out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img


# In[92]:


pip install ptflops


# In[93]:


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info


# In[94]:


with torch.cuda.device(0):
  net = models.densenet161()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[95]:


import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()


# In[97]:


from PIL import Image
from torchvision import transforms
input_image = Image.open('desktop/ML/bobby.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[98]:


if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


# In[99]:


with torch.no_grad():
    output = model(input_batch)


# In[100]:


print(output[0])


# In[101]:


probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[102]:


with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# In[103]:


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[106]:


input_image = Image.open('desktop/ML/eagle.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)
with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
    


# In[110]:


input_image2 = Image.open('desktop/ML/bobby.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[111]:


input_tensor = preprocess(input_image2)
input_batch = input_tensor.unsqueeze(0)
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)
with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[113]:


from PIL import Image
from torchvision import transforms
input_image = Image.open('desktop/ML/gator.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[114]:


if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


# In[115]:


with torch.no_grad():
    output = model(input_batch)


# In[116]:


with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# In[117]:


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[118]:


input_image = Image.open('desktop/ML/cat.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[119]:


if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')


# In[120]:


with torch.no_grad():
    output = model(input_batch)


# In[121]:


with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# In[122]:


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[123]:


percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage2[index[0]].item()


# In[124]:


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[125]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[126]:


_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[127]:


get_ipython().system('wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')


# In[128]:


input_image = Image.open('desktop/ML/cat.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[130]:


input_image


# In[131]:


print(output[0])


# In[132]:


probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[133]:


with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# In[134]:


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[135]:


input_image = Image.open('desktop/ML/eagle.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[136]:


probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[137]:


with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# In[138]:


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[139]:


print(output[0])


# In[140]:


input_image = Image.open('desktop/ML/eagle.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[141]:


input_image


# In[142]:


print(output[0])


# In[143]:


input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[144]:


print(output[0])


# In[145]:


input_image = Image.open('desktop/ML/eagle.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[146]:


input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


# In[147]:


print(output[0])


# In[148]:


probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# In[149]:


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# In[150]:


with open("desktop/ML/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# In[151]:


top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# In[152]:


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = models.densenet161()
  macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:




