import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),                    # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))      # Normalize pixel values between -1 and 1
])

mnist = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset=mnist, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x.view(-1, 28*28))

generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

epochs = 10

for epoch in range(epochs):
    for batch, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

                # Train Discriminator
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, 100)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

                # Train Generator
        z = torch.randn(batch_size, 100)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

with torch.no_grad():
    z = torch.randn(16, 100)
    generated = generator(z)
    generated = generated.view(-1, 1, 28, 28)

    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(generated[i][0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
