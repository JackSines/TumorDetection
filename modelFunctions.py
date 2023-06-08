import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
import numpy as np

TRAIN_DATASET_DIR = "dataset/Training"
TESTING_DATASET_DIR = "dataset/Testing"


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

    batch_size = 32

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

        for inputs, labels in train_loader:
            print("processing image...")
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

        epoch_loss = running_loss / len(trainingDataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

        # Validation
