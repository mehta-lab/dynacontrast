import numpy as np
import torch as t
from torch import nn as nn
from torch.nn import functional as F


CHANNEL_VAR = np.array([1., 1.])
CHANNEL_MAX = 65535.
eps = 1e-9


class VectorQuantizer(nn.Module):
    """ Vector Quantizer module as introduced in
        "Neural Discrete Representation Learning"

    This module contains a list of trainable embedding vectors, during training
    and inference encodings of inputs will find their closest resemblance
    in this list, which will be reassembled as quantized encodings (decoder
    input)

    """
    def __init__(self, embedding_dim=128, num_embeddings=128, commitment_cost=0.25, device='cuda:0'):
        """ Initialize the module

        Args:
            embedding_dim (int, optional): size of embedding vector
            num_embeddings (int, optional): number of embedding vectors
            commitment_cost (float, optional): balance between latent losses
            device (str, optional): device the model will be running on

        """
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.device = device
        self.w = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        """ Forward pass

        Args:
            inputs (torch tensor): encodings of input image

        Returns:
            torch tensor: quantized encodings (decoder input)
            torch tensor: quantization loss
            torch tensor: perplexity, measuring coverage of embedding vectors

        """
        # inputs: Batch * Num_hidden(=embedding_dim) * H * W
        distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)

        # Decoder input
        encoding_indices = t.argmax(-distances, 1)
        quantized = self.w(encoding_indices).transpose(2, 3).transpose(1, 2)
        assert quantized.shape == inputs.shape
        output_quantized = inputs + (quantized - inputs).detach()

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Perplexity (used to monitor)
        encoding_onehot = t.zeros(encoding_indices.flatten().shape[0], self.num_embeddings).to(self.device)
        encoding_onehot.scatter_(1, encoding_indices.flatten().unsqueeze(1), 1)
        avg_probs = t.mean(encoding_onehot, 0)
        perplexity = t.exp(-t.sum(avg_probs*t.log(avg_probs + 1e-10)))

        return output_quantized, loss, perplexity

    @property
    def embeddings(self):
        return self.w.weight

    def encode_inputs(self, inputs):
        """ Find closest embedding vector combinations of input encodings

        Args:
            inputs (torch tensor): encodings of input image

        Returns:
            torch tensor: index tensor of embedding vectors

        """
        # inputs: Batch * Num_hidden(=embedding_dim) * H * W
        distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)
        encoding_indices = t.argmax(-distances, 1)
        return encoding_indices

    def decode_inputs(self, encoding_indices):
        """ Assemble embedding vector index to quantized encodings

        Args:
            encoding_indices (torch tensor): index tensor of embedding vectors

        Returns:
            torch tensor: quantized encodings (decoder input)

        """
        quantized = self.w(encoding_indices).transpose(2, 3).transpose(1, 2)
        return quantized


class Reparametrize(nn.Module):
    """ Reparameterization step in RegularVAE
    """
    def forward(self, z_mean, z_logstd):
        """ Forward pass

        Args:
            z_mean (torch tensor): latent vector mean
            z_logstd (torch tensor): latent vector std (log)

        Returns:
            torch tensor: reparameterized latent vector
            torch tensor: KL divergence

        """
        z_std = t.exp(0.5 * z_logstd)
        eps = t.randn_like(z_std)
        z = z_mean + z_std * eps
        KLD = -0.5 * t.sum(1 + z_logstd - z_mean.pow(2) - z_logstd.exp())
        return z, KLD


class Reparametrize_IW(nn.Module):
    """ Reparameterization step in IWAE
    """
    def __init__(self, k=5, **kwargs):
        """ Initialize the module

        Args:
            k (int, optional): number of sampling trials
            **kwargs: other keyword arguments

        """
        super(Reparametrize_IW, self).__init__(**kwargs)
        self.k = k

    def forward(self, z_mean, z_logstd):
        """ Forward pass

        Args:
            z_mean (torch tensor): latent vector mean
            z_logstd (torch tensor): latent vector std (log)

        Returns:
            torch tensor: reparameterized latent vectors
            torch tensor: randomness

        """
        z_std = t.exp(0.5 * z_logstd)
        epss = [t.randn_like(z_std) for _ in range(self.k)]
        zs = [z_mean + z_std * eps for eps in epss]
        return zs, epss


class Flatten(nn.Module):
    """ Helper module for flatten tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(nn.Module):
    """ Customized residual block in network
    """
    def __init__(self,
                 num_hiddens=128,
                 num_residual_hiddens=512,
                 num_residual_layers=2):
        """ Initialize the module

        Args:
            num_hiddens (int, optional): number of hidden units
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers

        """
        super(ResidualBlock, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens

        self.layers = []
        for _ in range(self.num_residual_layers):
            self.layers.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.num_hiddens, self.num_residual_hiddens, 3, padding=1),
                nn.BatchNorm2d(self.num_residual_hiddens),
                nn.ReLU(),
                nn.Conv2d(self.num_residual_hiddens, self.num_hiddens, 1),
                nn.BatchNorm2d(self.num_hiddens)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """ Forward pass

        Args:
            x (torch tensor): input tensor

        Returns:
            torch tensor: output tensor

        """
        output = x
        for i in range(self.num_residual_layers):
            output = output + self.layers[i](output)
        return output



class VQ_VAE_z16(nn.Module):
    """ Reduced Vector-Quantized VAE with 16 X 16 X num_hiddens latent tensor
    """

    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 num_embeddings=64,
                 commitment_cost=0.25,
                 channel_var=CHANNEL_VAR,
                 weight_recon=1.,
                 weight_commitment=1.,
                 weight_matching=0.005,
                 device="cuda:0",
                 w_a=1.1,
                 w_t=0.1,
                 w_n=-0.5,
                 margin=0.5,
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            num_embeddings (int, optional): number of VQ embedding vectors
            commitment_cost (float, optional): balance between latent losses
            channel_var (list of float, optional): each channel's SD, used for
                balancing loss across channels
            weight_recon (float, optional): balance of reconstruction loss
            weight_commitment (float, optional): balance of commitment loss
            weight_matching (float, optional): balance of matching loss
            device (str, optional): device the model will be running on
            **kwargs: other keyword arguments

        """
        super(VQ_VAE_z16, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)),
                                        requires_grad=False)
        self.weight_recon = weight_recon
        self.weight_commitment = weight_commitment
        self.weight_matching = weight_matching
        self.w_a = w_a
        self.w_t = w_t
        self.w_n = w_n
        self.margin = margin
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
        self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost, device=device)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given,
                pairwise relationship between samples in the minibatch, used
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask
                of training samples, used to concentrate loss on cell bodies

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_after, c_loss, perplexity = self.vq(z_before)
        decoded = self.dec(z_after)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none') / self.channel_var)
        total_loss = self.weight_recon * recon_loss + self.weight_commitment * c_loss
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_before_ = z_before.reshape((z_before.shape[0], -1))
            len_latent = z_before_.shape[1]
            sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                            z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
            time_matching_weights = time_matching_mat.clone()
            time_matching_weights[time_matching_mat == 2] = self.w_a
            time_matching_weights[time_matching_mat == 1] = self.w_t
            time_matching_weights[time_matching_mat == 0] = self.w_n
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = sim_mat * time_matching_weights.clone()
            time_matching_loss[time_matching_mat == 0] = \
                t.clamp(time_matching_loss.clone()[time_matching_mat == 0] + self.margin, min=0)
            time_matching_loss = time_matching_loss.mean()
            total_loss += self.weight_matching * time_matching_loss
        return decoded, \
               {'recon_loss': recon_loss,
                'commitment_loss': c_loss,
                'time_matching_loss': time_matching_loss,
                'perplexity': perplexity,
                'total_loss': total_loss, }

    def predict(self, inputs):
        """ Prediction fn, same as forward pass """
        return self.forward(inputs)

class VQ_VAE_z32(nn.Module):
    """ Vector-Quantized VAE with 32 X 32 X num_hiddens latent tensor
     as introduced in  "Neural Discrete Representation Learning"
    """
    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 num_embeddings=64,
                 commitment_cost=0.25,
                 channel_var=np.ones(2),
                 weight_matching=0.005,
                 w_a=1.1,
                 w_t=0.1,
                 w_n=-0.5,
                 margin=0.5,
                 extra_loss=None,
                 device="cuda:0",
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            num_embeddings (int, optional): number of VQ embedding vectors
            commitment_cost (float, optional): balance between latent losses
            channel_var (list of float, optional): each channel's SD, used for
                balancing loss across channels
            alpha (float, optional): balance of matching loss
            extra_loss (None or dict): extra loss to add to the VQVAE loss
            with format {loss name: loss}.
            gpu (bool, optional): if the model is run on gpu
            **kwargs: other keyword arguments

        """
        super(VQ_VAE_z32, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
        self.weight_matching = weight_matching
        self.w_a = w_a
        self.w_t = w_t
        self.w_n = w_n
        self.margin = margin
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens // 2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens // 2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
        self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost, device=device)
        self.dec = nn.Sequential(
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens // 2, self.num_inputs, 4, stride=2, padding=1))
        self.extra_loss = extra_loss

    def forward(self, inputs, labels=None, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given,
                pairwise relationship between samples in the minibatch, used
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask
                of training samples, used to concentrate loss on cell bodies

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_after, c_loss, perplexity = self.vq(z_before)
        decoded = self.dec(z_after)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var)
        total_loss = recon_loss + c_loss
        #TODO: refactor to make time matching loss a class and pass it as extra loss argument
        time_matching_loss = 0
        if not time_matching_mat is None:
            z_after_ = z_after.reshape((z_after.shape[0], -1))
            len_latent = z_after_.shape[1]
            sim_mat = t.pow(z_after_.reshape((1, -1, len_latent)) - \
                            z_after_.reshape((-1, 1, len_latent)), 2).mean(2)
            time_matching_weights = time_matching_mat.clone()
            time_matching_weights[time_matching_mat == 2] = self.w_a
            time_matching_weights[time_matching_mat == 1] = self.w_t
            time_matching_weights[time_matching_mat == 0] = self.w_n
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = sim_mat * time_matching_weights.clone()
            time_matching_loss[time_matching_mat == 0] = \
                t.clamp(time_matching_loss.clone()[time_matching_mat == 0] + self.margin, min=0)
            time_matching_loss = time_matching_loss.mean()
            total_loss += time_matching_loss * self.weight_matching
        loss_dict = {'recon_loss': recon_loss,
                 'commitment_loss': c_loss,
                 'time_matching_loss': time_matching_loss,
                 'perplexity': perplexity,
                 'total_loss': total_loss, }
        if self.extra_loss is not None:
            z_after_ = z_after.reshape((z_after.shape[0], -1))
            for loss_name, loss_fn in self.extra_loss.items():
                extra_loss, frac_pos = loss_fn(labels, z_after_)
                total_loss += extra_loss * self.alpha
                loss_dict['total_loss'] = total_loss
                loss_dict[loss_name] = extra_loss
        return decoded, loss_dict

    def predict(self, inputs):
        """ Prediction fn, same as forward pass """
        return self.forward(inputs)


class VAE(nn.Module):
    """ Regular VAE """
    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 channel_var=CHANNEL_VAR,
                 weight_recon=1.,
                 weight_kld=1.,
                 weight_matching=0.005,
                 w_a=1.1,
                 w_t=0.1,
                 w_n=-0.5,
                 margin=0.5,
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            channel_var (list of float, optional): each channel's SD, used for
                balancing loss across channels
            weight_recon (float, optional): balance of reconstruction loss
            weight_kld (float, optional): balance of KL divergence
            weight_matching (float, optional): balance of matching loss
            **kwargs: other keyword arguments

        """
        super(VAE, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
        self.weight_recon = weight_recon
        self.weight_kld = weight_kld
        self.weight_matching = weight_matching
        self.w_a = w_a
        self.w_t = w_t
        self.w_n = w_n
        self.margin = margin
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
            nn.Conv2d(self.num_hiddens, 2*self.num_hiddens, 1))
        self.rp = Reparametrize()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given,
                pairwise relationship between samples in the minibatch, used
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask
                of training samples, used to concentrate loss on cell bodies

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_mean = z_before[:, :self.num_hiddens]
        z_logstd = z_before[:, self.num_hiddens:]

        # Reparameterization trick
        z_after, KLD = self.rp(z_mean, z_logstd)

        decoded = self.dec(z_after)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.sum(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var)
        total_loss = self.weight_recon * recon_loss + self.weight_kld * KLD
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_before_ = z_mean.reshape((z_mean.shape[0], -1))
            len_latent = z_before_.shape[1]
            sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                            z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
            time_matching_weights = time_matching_mat.clone()
            time_matching_weights[time_matching_mat == 2] = self.w_a
            time_matching_weights[time_matching_mat == 1] = self.w_t
            time_matching_weights[time_matching_mat == 0] = self.w_n
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = sim_mat * time_matching_weights.clone()
            time_matching_loss[time_matching_mat == 0] = \
                t.clamp(time_matching_loss.clone()[time_matching_mat == 0] + self.margin, min=0)
            time_matching_loss = time_matching_loss.mean()
            total_loss += self.weight_matching * time_matching_loss
        return decoded, \
               {'recon_loss': recon_loss/(inputs.shape[0] * 32768),
                'KLD': KLD,
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': t.zeros(())}

    def predict(self, inputs):
        """ Prediction fn without reparameterization

        Args:
            inputs (torch tensor): input cell image patches

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: reconstruction loss

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_mean = z_before[:, :self.num_hiddens]
        decoded = self.dec(z_mean)
        recon_loss = t.mean(F.mse_loss(decoded, inputs, reduction='none')/self.channel_var)
        return decoded, {'recon_loss': recon_loss}


class IWAE(VAE):
    """ Importance Weighted Autoencoder as introduced in
        "Importance Weighted Autoencoders"
    """
    def __init__(self, k=5, **kwargs):
        """ Initialize the model

        Args:
            k (int, optional): number of sampling trials
            **kwargs: other keyword arguments (including arguments for `VAE`)

        """
        super(IWAE, self).__init__(**kwargs)
        self.k = k
        self.rp = Reparametrize_IW(k=self.k)

    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given,
                pairwise relationship between samples in the minibatch, used
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask
                of training samples, used to concentrate loss on cell bodies

        Returns:
            None: placeholder
            dict: losses and perplexity of the minibatch

        """
        z_before = self.enc(inputs)
        z_mean = z_before[:, :self.num_hiddens]
        z_logstd = z_before[:, self.num_hiddens:]
        z_afters, epss = self.rp(z_mean, z_logstd)

        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_before_ = z_mean.reshape((z_mean.shape[0], -1))
            len_latent = z_before_.shape[1]
            sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                            z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
            time_matching_weights = time_matching_mat.clone()
            time_matching_weights[time_matching_mat == 2] = self.w_a
            time_matching_weights[time_matching_mat == 1] = self.w_t
            time_matching_weights[time_matching_mat == 0] = self.w_n
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = sim_mat * time_matching_weights.clone()
            time_matching_loss[time_matching_mat == 0] = \
                t.clamp(time_matching_loss.clone()[time_matching_mat == 0] + self.margin, min=0)
            time_matching_loss = time_matching_loss.mean()

        log_ws = []
        recon_losses = []
        for z, eps in zip(z_afters, epss):
            decoded = self.dec(z)
            log_p_x_z = - t.sum(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var, dim=(1, 2, 3))
            log_p_z = - t.sum(0.5 * z ** 2, dim=(1, 2, 3)) #- 0.5 * t.numel(z[0]) * np.log(2 * np.pi)
            log_q_z_x = - t.sum(0.5 * eps ** 2 + z_logstd, dim=(1, 2, 3)) #- 0.5 * t.numel(z[0]) * np.log(2 * np.pi)
            log_w_unnormed = log_p_x_z  + log_p_z - log_q_z_x
            log_ws.append(log_w_unnormed)
            recon_losses.append(-log_p_x_z)
        log_ws = t.stack(log_ws, 1)
        log_ws_minus_max = log_ws - t.max(log_ws, dim=1, keepdim=True)[0]
        ws = t.exp(log_ws_minus_max)
        normalized_ws = ws / t.sum(ws, dim=1, keepdim=True)
        loss = -(normalized_ws.detach() * log_ws).sum()
        total_loss = loss + self.weight_matching * time_matching_loss

        recon_losses = t.stack(recon_losses, 1)
        recon_loss = (normalized_ws.detach() * recon_losses).sum()
        return None, \
               {'recon_loss': recon_loss/(inputs.shape[0] * 32768),
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': t.zeros(())}


class AAE(nn.Module):
    """ Adversarial Autoencoder as introduced in
        "Adversarial Autoencoders"
    """
    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 channel_var=CHANNEL_VAR,
                 weight_recon=1.,
                 weight_matching=0.005,
                 w_a=1.1,
                 w_t=0.1,
                 w_n=-0.5,
                 margin=0.5,
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            channel_var (list of float, optional): each channel's SD, used for
                balancing loss across channels
            weight_recon (float, optional): balance of reconstruction loss
            weight_matching (float, optional): balance of matching loss
            **kwargs: other keyword arguments

        """
        super(AAE, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
        self.weight_recon = weight_recon
        self.weight_matching = weight_matching
        self.w_a = w_a
        self.w_t = w_t
        self.w_n = w_n
        self.margin = margin
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
        self.enc_d = nn.Sequential(
            nn.Conv2d(self.num_hiddens, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(self.num_hiddens * 2, self.num_hiddens * 8),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(self.num_hiddens * 8, self.num_hiddens),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, 1),
            nn.Sigmoid())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given,
                pairwise relationship between samples in the minibatch, used
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask
                of training samples, used to concentrate loss on cell bodies

        Returns:
            None: placeholder
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z = self.enc(inputs)
        decoded = self.dec(z)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var)
        total_loss = self.weight_recon * recon_loss
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_ = z.reshape((z.shape[0], -1))
            len_latent = z_.shape[1]
            sim_mat = t.pow(z_.reshape((1, -1, len_latent)) - \
                            z_.reshape((-1, 1, len_latent)), 2).mean(2)
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_weights = time_matching_mat.clone()
            time_matching_weights[time_matching_mat == 2] = self.w_a
            time_matching_weights[time_matching_mat == 1] = self.w_t
            time_matching_weights[time_matching_mat == 0] = self.w_n
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = sim_mat * time_matching_weights.clone()
            time_matching_loss[time_matching_mat == 0] = \
                t.clamp(time_matching_loss.clone()[time_matching_mat == 0] + self.margin, min=0)
            time_matching_loss = time_matching_loss.mean()
            total_loss += self.weight_matching * time_matching_loss
        return decoded, \
               {'recon_loss': recon_loss,
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': t.zeros(())}

    def adversarial_loss(self, inputs):
        """ Calculate adversarial loss for the batch

        Args:
            inputs (torch tensor): input cell image patches

        Returns:
            dict: generator/discriminator losses

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_data = self.enc(inputs)
        z_prior = t.randn_like(z_data)
        _z_data = self.enc_d(z_data)
        _z_prior = self.enc_d(z_prior)
        g_loss = -t.mean(t.log(_z_data + eps))
        d_loss = -t.mean(t.log(_z_prior + eps) + t.log(1 - _z_data.detach() + eps))
        return {'generator_loss': g_loss,
                'descriminator_loss': d_loss,
                'score': t.mean(_z_data)}

    def predict(self, inputs):
        """ Prediction fn, same as forward pass """
        return self.forward(inputs)