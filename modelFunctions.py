import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
import numpy as np

TRAIN_DATASET_DIR = "dataset_full/Training"
TESTING_DATASET_DIR = "dataset_full/Testing"


def calc_mean_std():
    # Transformations for each image - this excludes normalisation as we're calculating the values now
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to tensors
    ])

    trainingDataset = datasets.ImageFolder(root=TRAIN_DATASET_DIR, transform=data_transform)

    # Iterate over the dataset and accumulate intensity values
    intensity_values = []
    for image, _ in trainingDataset:
        intensity_values.append(np.mean(image.numpy()))

    # Calculate the mean and standard deviation
    mean_intensity = np.mean(intensity_values)
    std_intensity = np.std(intensity_values)

    # Define the path to the output text file
    output_path = 'output.txt'

    # Open the text file in write mode
    with open(output_path, 'w') as txt_file:
        # Write the mean and standard deviation to the file
        txt_file.write("Mean: {}\n".format(mean_intensity))
        txt_file.write("Standard Deviation: {}\n".format(std_intensity))


def train():
    print("Preparing dataset...")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.185], std=[0.069])  # Normalise data
    ])

    trainingDataset = datasets.ImageFolder(root=TRAIN_DATASET_DIR, transform=data_transform)

    testDataset = datasets.ImageFolder(root=TESTING_DATASET_DIR, transform=data_transform)

    batch_size = 32 #32

    train_loader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size)
    # Prepare model #
    print("Preparing model...")
    # Image input size
    input_size = 224

    # Load pre-trained model
    model = EfficientNet.from_pretrained('efficientnet-b5')

    # Modify last layer
    num_classes = 4
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_classes)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training loop
    num_epochs = 5
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train() # Set to training mode
        running_loss = 0.0
        print('Epoch [{}/{}]'.format(epoch.__index__()+1, num_epochs))
        image_counter = 1
        for inputs, labels in train_loader:
            print('processing batch [{}/{}]'.format(image_counter, (len(trainingDataset)/batch_size)))
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimiser.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimisation
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * inputs.size(0)
            image_counter += 1

        epoch_loss = running_loss / len(trainingDataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

        # Save the weights after every 1 epoch
        if epoch % 1 == 0:
            torch.save(model.state_dict(), 'model_weights_epoch_{}.pth'.format(epoch))

        # Validation
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_corrects = 0

        # Create empty lists to store validation metrics
        val_loss_values = []
        val_acc_values = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

            val_loss = val_loss / len(test_loader)
            val_acc = val_corrects.double() / len(test_loader)
            # Save validation metrics
            val_loss_values.append(val_loss)
            val_acc_values.append(val_acc.item())

            # Define the path to the output text file
            output_path = 'val{}.txt'.format(epoch)

            # Open the text file in write mode
            with open(output_path, 'w') as txt_file:
                # Write the mean and standard deviation to the file
                txt_file.write("Loss: {}\n".format(val_loss))
                txt_file.write("Acc: {}\n".format(val_acc))

            print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(val_loss, val_acc))

def trainingGraphs():
    import matplotlib.pyplot as plt

    num_epochs = 5
    files = ['val{}.txt'.format(i) for i in range(num_epochs)]

    epochs = []
    loss_values = []
    accuracy_values = []

    # Read loss data
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('Loss:'):
                    loss = float(line.split()[1])
                    loss_values.append(loss)
                if line.startswith('Acc:'):
                    accuracy = float(line.split()[1])
                    accuracy_values.append(accuracy)

    # Create epoch list
    epochs = list(range(1, num_epochs + 1))

    # Plot loss over epochs
    plt.figure()
    plt.plot(epochs, loss_values, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

    # Plot accuracy over epochs
    plt.figure()
    plt.plot(epochs, accuracy_values, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.show()
