%reload_ext autoreload
%autoreload 2

import torch
import torch.optim as optim

from my_datasets.reid_dataset import Market1501

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils.split_data import ValidationSplitter, TrainingSplitter
from utils.image_utils import imshow

from models.reid_model import ReIdModel, LeNet
import cost_functions
from torch.utils.tensorboard import SummaryWriter

TRAIN_ROOT = "/media/deayalar/Data/Documents/Unitn/Deep Learning/Assignment/dataset/train"

# This validation set is used to estimate the performance on the final test set
splitter = ValidationSplitter(train_root=TRAIN_ROOT)
train_set, val_estimation_set = splitter.split(train_size=0.75, random_seed=42)

# Create a validation set for training
train_set, val_set = TrainingSplitter().split(train_set, train_size=0.8, random_seed=42)

#Create pytorch Datasets
composed = transforms.Compose([transforms.ToTensor()])

train_dataset = Market1501(root_dir=TRAIN_ROOT, 
                            images_list=train_set,
                            transform=composed)
val_dataset = Market1501(root_dir=TRAIN_ROOT, 
                         images_list=val_set,
                         transform=composed)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

def get_optimizer(net, lr, wd, momentum):
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
  return optimizer

def test(net, data_loader, cost_function, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.

  # Strictly needed if network contains layers which has different behaviours between train and test
  net.eval()
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader):
      # Load data into GPU
      inputs = inputs.to(device)
      targets = targets.to(device)
        
      # Forward pass
      outputs = net(inputs)

      # Apply the loss
      loss = cost_function(outputs, targets)

      # Better print something
      samples += inputs.shape[0]
      cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
      _, predicted = outputs.max(1)
      cumulative_accuracy += predicted.eq(targets).sum().item()

  return cumulative_loss/samples, cumulative_accuracy/samples*100


# TRAINING
def train(model, data_loader, optimizer, cost_function, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.

  # Strictly needed if network contains layers which has different behaviours between train and test
  model.train()
  for batch_idx, (inputs, targets) in enumerate(data_loader):
    # Load data into GPU
    inputs = inputs.to(device)
    targets = targets.to(device)
      
    # Forward pass
    outputs = model(inputs)

    # Apply the loss
    loss = cost_function(outputs, targets)
      
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Zeros the gradient
    optimizer.zero_grad()

    # Better print something, no?
    samples += inputs.shape[0]
    cumulative_loss += loss.item()
    _, predicted = outputs.max(1)
    cumulative_accuracy += predicted.eq(targets).sum().item()

  return cumulative_loss/samples, cumulative_accuracy/samples*100


def main(batch_size=128, 
         device='cuda:0', 
         learning_rate=0.01, 
         weight_decay=0.000001, 
         momentum=0.9, 
         epochs=1, 
         visualization_name='lenet',
         dataset='mnist', 
         norm=False):
  
  # Creates a logger for the experiment
  writer = SummaryWriter(log_dir=f"runs/{visualization_name}")

  #train_loader, val_loader, test_loader = get_data(batch_size=batch_size, 
  #                                                 test_batch_size=batch_size, 
  #                                                 dataset=dataset)
  
  net = LeNet().to(torch.device(device))
  
  optimizer =  get_optimizer(net, learning_rate, weight_decay, momentum)
  
  cost_function = cost_functions.cross_entropy()

  print('Before training:')
  train_loss, train_accuracy = test(net, train_loader, cost_function)
  val_loss, val_accuracy = test(net, val_loader, cost_function)
  #test_loss, test_accuracy = test(net, test_loader, cost_function)

  print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
  print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
  #print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
  print('-----------------------------------------------------')
  
  # Add values to plots
  writer.add_scalar('Loss/train_loss', train_loss, 0)
  writer.add_scalar('Loss/val_loss', val_loss, 0)
  writer.add_scalar('Accuracy/train_accuracy', train_accuracy, 0)
  writer.add_scalar('Accuracy/val_accuracy', val_accuracy, 0)

  for e in range(epochs):
    train_loss, train_accuracy = train(net, train_loader, optimizer, cost_function)
    val_loss, val_accuracy = test(net, val_loader, cost_function)
    print('Epoch: {:d}'.format(e+1))
    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
    print('-----------------------------------------------------')
    
    # Add values to plots
    writer.add_scalar('Loss/train_loss', train_loss, e + 1)
    writer.add_scalar('Loss/val_loss', val_loss, e + 1)
    writer.add_scalar('Accuracy/train_accuracy', train_accuracy, e + 1)
    writer.add_scalar('Accuracy/val_accuracy', val_accuracy, e + 1)

  print('After training:')
  train_loss, train_accuracy = test(net, train_loader, cost_function)
  val_loss, val_accuracy = test(net, val_loader, cost_function)
  #test_loss, test_accuracy = test(net, test_loader, cost_function)

  print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
  print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
  #print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
  print('-----------------------------------------------------')

  # Closes the logger
  writer.close()


main()

# VALIDATION

val_estimation_dataset = Market1501(root_dir=TRAIN_ROOT, 
                               images_list=val_estimation_set,
                               transform=composed)