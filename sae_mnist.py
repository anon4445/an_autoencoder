import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchsummary import summary
import os 
import certifi

# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class SparseAutoencoder(nn.Module):
    """
    A Sparse Autoencoder with:
        - Encoder: Linear -> Sigmoid
        - Decoder: Linear -> Tanh
        - Sparsity Penalty: KL Divergence
        - Combined loss: MSE + sparsity penalty
    """
    def __init__(self, in_dims, h_dims, sparsity_lambda=1e-4, sparsity_target=0.05, xavier_norm_init=True):
        super().__init__()
        self.in_dims = in_dims
        self.h_dims = h_dims
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target
        self.xavier_norm_init = xavier_norm_init

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dims, self.h_dims),
            nn.Sigmoid()
        )

        # (Optional) Xavier initialization for encoder
        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.encoder[0].weight)
            nn.init.constant_(self.encoder[0].bias, 0)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dims, self.in_dims),
            nn.Tanh()
        )

        # (Optional) Xavier initialization for decoder
        if self.xavier_norm_init:
            nn.init.xavier_uniform_(self.decoder[0].weight)
            nn.init.constant_(self.decoder[0].bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def sparsity_penalty(self, encoded):
        """
        Computes the KL-divergence-based penalty to enforce a desired sparsity level (rho).
        """
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.sparsity_target
        epsilon = 1e-8

        # Ensure numerical stability
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)

        # KL Divergence
        kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        sparsity_penalty = torch.sum(kl_div)

        return self.sparsity_lambda * sparsity_penalty

    def loss_function(self, x_hat, x, encoded):
        """
        Combines MSE loss for reconstruction and the KL-divergence-based sparsity penalty.
        """
        mse_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_penalty(encoded)
        return mse_loss + sparsity_loss


def train_model(model, train_loader, val_loader, n_epochs, optimizer, device):
    model.to(device)
    train_loss_history = []
    val_loss_history = []

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()

            encoded, decoded = model(data)
            loss = model.loss_function(decoded, data, encoded)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.view(data.size(0), -1).to(device)
                encoded, decoded = model(data)
                loss = model.loss_function(decoded, data, encoded)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    print('Training complete.')

    return train_loss_history, val_loss_history


def plot_loss(train_loss, val_loss, save_path=None):
    """
    Plots both train and validation loss over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', linestyle='-', label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker='s', linestyle='--', label='Validation Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=600)
        print(f'Loss plot saved to {save_path}.')

    plt.show()


if __name__ == "__main__":

    # Set random seed for reproducibility
    def seeding(seed):
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seeding(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--in_dims', type=int, default=784)
    parser.add_argument('--h_dims', type=int, default=256)
    parser.add_argument('--sparsity_lambda', type=float, default=1e-4)
    parser.add_argument('--sparsity_target', type=float, default=0.05)
    parser.add_argument('--xavier_norm_init', type=bool, default=True)
    parser.add_argument('--download_mnist', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_plot', type=bool, default=True)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=args.download_mnist)

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    sae_model = SparseAutoencoder(
        in_dims=args.in_dims, 
        h_dims=args.h_dims, 
        sparsity_lambda=args.sparsity_lambda, 
        sparsity_target=args.sparsity_target,
        xavier_norm_init=args.xavier_norm_init
    )

    optimizer = torch.optim.Adam(sae_model.parameters(), lr=args.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using [{str(device).upper()}] for training.')

    if args.train:
        print('\nTraining...')
        train_loss, val_loss = train_model(
            model=sae_model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=args.n_epochs,
            optimizer=optimizer,
            device=device
        )

        # Plot train & validation loss
        plot_loss(train_loss, val_loss, save_path='./loss_plot.png' if args.save_plot else None)

    if args.save_model:
        os.makedirs('./files', exist_ok=True)
        torch.save(sae_model.state_dict(), './files/sae_model.pth')
        print('Model saved to ./files/sae_model.pth')