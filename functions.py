import time
import copy
import torch
import numpy as np
from PIL import Image
from torchvision import models

def train_model(model, criterion, optimizer, scheduler, num_epochs, device, dataloaders, dataset_sizes):
    since = time.time()
    print('Training model, please wait.')
    
    model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# TODO: Do validation on the test set
def test_model(model, dataloaders, data_type='test', device='cuda'):
    print('Testing model, please wait.')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders[data_type]:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('total number of images = {}'.format(total))
    print('number of correct predictions = {}'.format(correct))
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
   
    network = checkpoint['network']
    if network == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif network == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif network == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif network == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print('Network {} not found'.format(checkpoint['network']))

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    if image.width < image.height:
        image.thumbnail((256,100000))
    else:
        image.thumbnail((100000,256))
        
    x1 = (image.width-224)/2
    y1 = (image.height-224)/2
    x2 = x1+224
    y2 = y1+224
    image = image.crop((x1, y1, x2, y2))
    
    image = np.array(image)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image-mean)/std
    
    image = image.transpose((2,0,1))
   
    return image

def predict(image_path, model, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    t_image = torch.from_numpy(image).type(torch.FloatTensor) 
    t_image.unsqueeze_(0)
    output = model.forward(t_image)
    probas = torch.exp(output)
    topk_probas, num_topk_labels = probas.topk(topk)
    
    topk_probas = topk_probas.data[0,:]
      
    num_topk_labels = num_topk_labels.data[0,:]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}    
    topk_classes = [cat_to_name[idx_to_class[index]] for index in num_topk_labels.numpy()]
    return topk_probas, topk_classes