from utils import (
    get_image_file_paths,
    pad_image,
    normalize_alpha_matte,
    normalize_array,
    depth_based_soft_foreground_pixel_visibility_map,
    display_image,
)
from mesh_utils import create_mesh, sample_novel_views
import os
from PIL import Image
from models import ImageCompletionModel,InpaintCAModel, Discriminator, matting_model, monocular_depth_model
import numpy as np
import argparse
from omegaconf import OmegaConf
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#import GPUtil

class ImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((400, 400))
        # padded_image, image_mask = pad_image(image)

        if self.transform:
            image = self.transform(image)
            # padded_image = self.transform(padded_image)
            # image_mask = th.from_numpy(image_mask).float()
        return image #, padded_image, image_mask

# Loss functions
def adversarial_loss(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true)

def hinge_loss_discriminator(y_pred, y_true):
    return th.mean(th.relu(1. - y_true * y_pred))

def hinge_loss_generator(y_pred):
    return -th.mean(y_pred)

def reconstruction_loss(y_pred, y_true):
    return nn.L1Loss()(y_pred, y_true)

def train_model(config):
    # Training configuration
    device = th.device(f"cuda:{config.training.gpu_id}" if th.cuda.is_available() else "cpu")
    epochs = config.training.epochs
    learning_rate = config.training.learning_rate
    save_interval = config.training.save_interval
    log_interval = config.training.log_interval

    # Datasets and DataLoaders
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(file_paths=config.data.train_image_paths, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)

    # Models
    generator = InpaintCAModel().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        running_loss_G = 0.0
        running_loss_D = 0.0

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            masks = th.randint(0, 2, (images.size(0), 1, images.size(2), images.size(3))).float().to(device)
            masked_images = images * (1 - masks)

            
            # Train Discriminator
            optimizer_D.zero_grad()
            alpha_matte = matting_model.get_alpha_matte(images)
            background_image = generator(images, alpha_matte)
            
            real_labels = th.ones(images.size(0), 1).to(device)
            fake_labels = th.zeros(images.size(0), 1).to(device)

            outputs_real = discriminator(images)
            outputs_fake = discriminator(background_image.detach())
            
            d_loss_real = hinge_loss_discriminator(outputs_real, real_labels)
            d_loss_fake = hinge_loss_discriminator(outputs_fake, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            background_image = generator(images, alpha_matte)
            outputs = discriminator(background_image)
            g_loss_adv = hinge_loss_generator(outputs)
            g_loss_rec = reconstruction_loss(background_image, images)

            g_loss = g_loss_adv + g_loss_rec
            g_loss.backward()
            optimizer_G.step()

            running_loss_G += g_loss.item()
            running_loss_D += d_loss.item()

            if batch_idx % log_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")

        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_D = running_loss_D / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average G Loss: {avg_loss_G:.4f}, Average D Loss: {avg_loss_D:.4f}")

        # Save the model at each save_interval epoch
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            if not os.path.exists(config.data.save_dir):
                os.makedirs(config.data.save_dir)
            th.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss_G': avg_loss_G,
                'loss_D': avg_loss_D,
            }, os.path.join(config.data.save_dir, f'model_epoch_{epoch+1}.pth'))

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    train_model(config)