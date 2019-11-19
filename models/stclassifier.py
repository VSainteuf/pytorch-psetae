import torch.nn as nn

from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder
from models.decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True, extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 positions=None,
                 mlp4=[128, 64, 32, 20]):
        super(PseTae, self).__init__()
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                       extra_size=extra_size)
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                n_neurons=mlp3, dropout=dropout,
                                                T=T, len_max_seq=len_max_seq, positions=positions)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])

    def forward(self, input):
        out = self.spatial_encoder(input)
        out = self.temporal_encoder(out)
        out = self.decoder(out)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL PARAMS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f} , Temporal {:5.1f}, Classifier {:5.1f}'.format(s / total * 100,
                                                                                      t / total * 100,
                                                                                      c / total * 100))

def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
