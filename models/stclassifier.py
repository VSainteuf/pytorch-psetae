import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.pse import PixelSetEncoder
from models.tae import TemporalAttentionEncoder
from models.decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4,
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
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        out = self.spatial_encoder(input)
        out = self.temporal_encoder(out)
        out = self.decoder(out)
        return out

    def param_ratio(self):
        total = get_ntrainparams(self)
        s = get_ntrainparams(self.spatial_encoder)
        t = get_ntrainparams(self.temporal_encoder)
        c = get_ntrainparams(self.decoder)

        print('TOTAL TRAINABLE PARAMETERS : {}'.format(total))
        print('RATIOS: Spatial {:5.1f}% , Temporal {:5.1f}% , Classifier {:5.1f}%'.format(s / total * 100,
                                                                                          t / total * 100,
                                                                                          c / total * 100))


class PseTae_pretrained(nn.Module):

    def __init__(self, weight_folder, hyperparameters, device='cuda', fold='all'):
        """
        Pretrained PseTea classifier.
        The class can either load the weights of a single fold or aggregate the predictions of the different sets of
        weights obtained during k-fold cross-validation and produces a single prediction.
        Args:
            weight_folder (str): Path to the folder containing the different sets of weights obtained during each fold
            (res_dir of the training script)
            hyperparameters (dict): Hyperparameters of the PseTae classifier
            device (str): Device on which the model should be loaded ('cpu' or 'cuda')
            fold( str or int): load all folds ('all') or number of the fold to load
        """
        super(PseTae_pretrained, self).__init__()
        self.weight_folder = weight_folder
        self.hyperparameters = hyperparameters

        self.fold_folders = [os.path.join(weight_folder, f) for f in os.listdir(weight_folder) if
                             os.path.isdir(os.path.join(weight_folder, f))]
        if fold == 'all':
            self.n_folds = len(self.fold_folders)
        else:
            self.n_folds = 1
            self.fold_folders = [self.fold_folders[int(fold) - 1]]
        self.model_instances = []

        print('Loading pre-trained models . . .')
        for f in self.fold_folders:
            m = PseTae(**hyperparameters)

            if device == 'cpu':
                map_loc = 'cpu'
            else:
                map_loc = 'cuda:{}'.format(torch.cuda.current_device())
                m = m.cuda()
            d = torch.load(os.path.join(f, 'model.pth.tar'), map_location=map_loc)
            m.load_state_dict(d['state_dict'])
            self.model_instances.append(m)
        print('Successfully loaded {} model instances'.format(self.n_folds))

    def forward(self, input):
        """ Returns class logits
        Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            outputs = [F.log_softmax(m(input), dim=1) for m in self.model_instances]
            outputs = torch.stack(outputs, dim=0).mean(dim=0)
        return outputs

    def predict_class(self, input):
        """Returns class prediction
                Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
                    Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
                    Pixel-Mask : Batch_size x Sequence length x Number of pixels
                    Extra-features : Batch_size x Sequence length x Number of features
        """
        with torch.no_grad():
            pred = self.forward(input).argmax(dim=1)
        return pred


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
