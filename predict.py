import argparse
import os
import sys
import json
import torch
import functions
from torchvision import datasets, transforms
from PIL import Image

def main(argv):

    parser = argparse.ArgumentParser(description='Predict flower name from an image.',
                                     prog='predict.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image_path', type=str, nargs=1, default=['flowers/test/1/image_06754.jpg'], help='Path to the image to use for prediction.')
    parser.add_argument('checkpoint', type=str, nargs=1, default=['checkpoint.pth'], help='Path to the checkpoint containing the model.')
    parser.add_argument('--top_k', type=int, nargs=1, default=[1], help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=open, nargs=1, default=['cat_to_name.json'], help='Use a mapping of categories to real names.')
    parser.add_argument('--gpu', nargs='?', const='cuda', help='Use GPU for inference.')
    args = vars(parser.parse_args(argv))
    image_path = args['image_path'][0]
    checkpoint = args['checkpoint'][0]
    topK = args['top_k'][0]
    category_names_file = args['category_names'][0]
    device = args['gpu']
    if device == None:
        device = 'cpu'
    
    print('parameters used for prediction:')
    print('-------------------------------')
    print('image_path = {}'.format(image_path))
    print('checkpoint = {}'.format(checkpoint))
    print('topK = {}'.format(topK))
    print('category_names_file = {}'.format(category_names_file))
    print('device = {}'.format(device))
    
    model = functions.load_checkpoint(checkpoint)
    model.eval() 
    model.to(device)
    
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
    image_datasets = {data_type: datasets.ImageFolder(os.path.join('flowers', data_type),
                                              data_transforms[data_type])
                      for data_type in ['train', 'valid', 'test']}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {data_type: torch.utils.data.DataLoader(image_datasets[data_type], batch_size=32,shuffle=True)
                  for data_type in ['train', 'valid', 'test']}
    
    # TODO: Display an image along with the top 5 classes    
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    flower_class = image_path.split('/')[2]
    title = cat_to_name[flower_class]

    probas, classes = functions.predict(image_path, model, cat_to_name, topk = topK) 

    print()
    print('image of {} has been predicted as:'.format(title))
    print('------------------------------------------')
    for index in range(len(classes)):
        print('{} : {:.2f}%'.format(classes[index], 100*probas[index]))

    
if __name__ == "__main__":
   main(sys.argv[1:])