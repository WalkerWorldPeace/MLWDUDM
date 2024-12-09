import torch
import torch.nn as nn
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Classifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=64):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.silu = nn.SiLU()
        self.batchnorm = nn.BatchNorm1d(output_dim, track_running_stats=False)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(output_dim*2, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
    def forward(self, x, t=None):
        if t is not None:
            emb = self.time_embed(timestep_embedding(t, self.output_dim*2))
            out = self.linear1(x+emb)
            out = self.batchnorm(out)
            out = self.silu(out)
            out = self.linear2(out)
        else:
            out = self.linear1(x)
            out = self.batchnorm(out)
            out = self.silu(out)
            out = self.linear2(out)

        return out

if __name__ == '__main__':
    t = torch.tensor([0,0])
    print(timestep_embedding(t, 64))
    model = Classifier()
    data = torch.randn(2, 64)
    x = model(data)
    print(x.shape)