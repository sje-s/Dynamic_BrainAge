from dynamic_brainage.models.bilstm import BiLSTM
from dynamic_brainage.models.gru import BiGRU
from dynamic_brainage.models.transformer import MyTransformer
from dynamic_brainage.models.convrnn import ConvRNN
from dynamic_brainage.models.conv1d import CNN1D
from dynamic_brainage.models.mlp import MLP
from dynamic_brainage.models.sequence_modeling.seq_rnn import SeqRNN
from dynamic_brainage.models.sequence_modeling.simple_rnn import SimpleSeqRNN
from dynamic_brainage.models.sequence_modeling.transformer.transformer import SeqTransformer
from dynamic_brainage.models.sequence_modeling.convrnn import SeqConvRNN
from dynamic_brainage.models.sequence_modeling.conv1d import SeqCNN1D
from dynamic_brainage.models.sequence_modeling.seqmlp import SeqMLP
from dynamic_brainage.models.conv2d import CNN2D
from dynamic_brainage.models.conv3d import CNN3D
from dynamic_brainage.models.bcgcn import BC_GCN, BC_GCN_Res, BC_GCN_SE, weights_init
from dynamic_brainage.models.rbcgcn import rBC_GCN
#from dynamic_brainage.models.resnet3d import ResNet_l0, ResNet_l1, ResNet_l2, ResNet_l3, ResNet_l4

def get_model(key, *args, **kwargs):
    if key.lower() == "bilstm":
        return BiLSTM(*args, **kwargs)
    elif key.lower() == "mlp":
        return MLP(*args, **kwargs)
    elif key.lower() == "seqmlp":
        return SeqMLP(*args, **kwargs)
    elif key.lower() == "bigru":
        return BiGRU(*args, **kwargs)
    elif key.lower() == 'transformer':
        return MyTransformer(*args, **kwargs)
    elif key.lower() == 'conv1d':
        return CNN1D(*args, **kwargs)
    elif key.lower() == "conv2d":
        return CNN2D(*args, **kwargs)
    elif key.lower() == "conv3d":
        return CNN3D(*args, **kwargs)
    elif key.lower() == 'convrnn':
        return ConvRNN(*args, **kwargs)
    elif key.lower() == 'convrnn_transpose':
        return ConvRNN(*args, channel=False, **kwargs)
    elif key.lower() == 'seqrnn':
        return SimpleSeqRNN(*args, **kwargs)
    elif key.lower() == 'seqformer':
        return SeqTransformer(*args, **kwargs)
    elif key.lower() == "seqconvrnn":
        return SeqConvRNN(*args, **kwargs)
    elif key.lower() == "seqconv1d":
        return SeqCNN1D(*args, **kwargs)
    elif key.lower() == "bcgcn":
        model = BC_GCN(*args, **kwargs)
        model.apply(weights_init)
        return model
    elif key.lower() == "bcgcn_res":
        model = BC_GCN_Res(*args, **kwargs)
        model.apply(weights_init)
        return model
    elif key.lower() == "bcgcn_se":
        model = BC_GCN_SE(*args, **kwargs)
        model.apply(weights_init) 
        return model
    elif key.lower() == "rbcgcn":
        model = rBC_GCN(*args, **kwargs)
        model.apply(weights_init) 
        return model