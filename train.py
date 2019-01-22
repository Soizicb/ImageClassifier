import os
import sys
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models    
from collections import OrderedDict
import functions
import argparse

def dir_path(string):
    print(string)
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def main(argv):
    parser = argparse.ArgumentParser(description='Train model to predict flower name from an image.',
                                     prog='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=dir_path, nargs=1, help='Path to the directory containing the images.')
    parser.add_argument('--save_dir', type=dir_path, nargs=1, default=['.'], help='Set directory to save checkpoints.')
    parser.add_argument('--arch', type=str, nargs=1, default=['vgg16'], help='Architecture.', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--learning_rate', type=float, nargs=1, default=[0.001], help='Learning rate.')
    parser.add_argument('--hidden_units', type=int, nargs=1, default=[512], help='Hidden units.')
    parser.add_argument('--epochs', type=int, nargs=1, default=[20], help='Number of epochs.')
    parser.add_argument('--gpu', nargs='?', const='cuda', help='Use GPU for inference.')
    args = parser.parse_args(argv)
    vargs = vars(args)
  
    data_dir = vargs['data_dir'][0]
    save_dir = vargs['save_dir'][0]
    arch = vargs['arch'][0]
    learning_rate = vargs['learning_rate'][0]
    epochs = vargs['epochs'][0]
    if epochs < 1:
        print('epochs must be an integer greater or equal to 1')
        sys.exit(2)
    hidden_units = vargs['hidden_units'][0]
    if hidden_units < 102:
        print('hidden_units must be an integer greater or equal to 102')
        sys.exit(2)
    device = vargs['gpu']
    if device == None:
        device = 'cpu'
        
    print('parameters used for training:')
    print('-----------------------------')     
    print("data_dir = {}".format(data_dir))
    print("save_dir={}".format(save_dir))
    print("arch={}".format(arch))
    print("learning_rate={}".format(learning_rate))
    print("hidden_units={}".format(hidden_units))
    print("epochs={}".format(epochs))
    print("device={}".format(device))
       
    train_dir = data_dir + '/train'
    if not os.path.isdir(train_dir):
         print('{} does not contain the mandatory train subdirectory'.format(data_dir))
         sys.exit(2)
    valid_dir = data_dir + '/valid'
    if not os.path.isdir(valid_dir):
         print('{} does not contain the mandatory valid subdirectory'.format(data_dir))
         sys.exit(2)
    test_dir = data_dir + '/test'
    if not os.path.isdir(test_dir):
         print('{} does not contain the mandatory test subdirectory'.format(data_dir))
         sys.exit(2)

   # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
                     'train': transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
                     'valid': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
                     'test': transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
                        }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {data_type: datasets.ImageFolder(os.path.join(data_dir, data_type),
                                              data_transforms[data_type])
                      for data_type in ['train', 'valid', 'test']}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {data_type: torch.utils.data.DataLoader(image_datasets[data_type], batch_size=32,shuffle=True)
                  for data_type in ['train', 'valid', 'test']}

    dataset_sizes = {data_type: len(image_datasets[data_type]) for data_type in ['train', 'valid', 'test']}

    # TODO: Build and train your network
    if arch == 'vgg11':
         model = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
          model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
         model = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
         model = models.vgg19(pretrained=True)
    else:
         print('Architecture {} not supported'.format(arch))
         print('Supported architectures are vgg11, vgg13, vgg16 and vgg19')
         sys.exit(2)
    for param in model.parameters():
         param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                               ('fc1', nn.Linear(25088, hidden_units)),
                               ('relu1', nn.ReLU()),
                               ('dropout1', nn.Dropout(0.2)),
                               ('fc2', nn.Linear(hidden_units, hidden_units)),
                               ('relu2', nn.ReLU()),
                               ('dropout2', nn.Dropout(0.2)),
                               ('fc3', nn.Linear(hidden_units, 102)),
                               ('output', nn.LogSoftmax(dim=1))
                               ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    functions.train_model(model, criterion, optimizer, scheduler, epochs, device, dataloaders, dataset_sizes)
    functions.test_model(model, dataloaders, device=device)

    # TODO: Save the checkpoint 
    model.class_to_idx = image_datasets['train'].class_to_idx
    #always saving on cpu to reuse without having to consume gpu time
    model.to('cpu')
    checkpoint2 = {'network': arch,
                   'classifier':model.classifier,
                   'state_dict': model.state_dict(),
                   'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint2, save_dir+'/checkpoint2.pth')


if __name__ == "__main__":
    main(sys.argv[1:])


