from dynamic_brainage.models.bilstm import BiLSTM
from dynamic_brainage.models.gru import BiGRU
from dynamic_brainage.models.transformer import MyTransformer
from dynamic_brainage.models.convrnn import ConvRNN
from dynamic_brainage.models.conv1d import CNN1D
from dynamic_brainage.models.sequence_modeling.seq_rnn import SeqRNN
from dynamic_brainage.models.sequence_modeling.simple_rnn import SimpleSeqRNN
from dynamic_brainage.models.sequence_modeling.transformer.transformer import SeqTransformer


def get_model(key, *args, **kwargs):
    if key.lower() == "bilstm":
        return BiLSTM(*args, **kwargs)
    elif key.lower() == "bigru":
        return BiGRU(*args, **kwargs)
    elif key.lower() == 'transformer':
        return MyTransformer(*args, **kwargs)
    elif key.lower() == 'conv1d':
        return CNN1D(*args, **kwargs)
    elif key.lower() == 'convrnn':
        return ConvRNN(*args, **kwargs)
    elif key.lower() == 'convrnn_transpose':
        return ConvRNN(*args, channel=False, **kwargs)
    elif key.lower() == 'seqrnn':
        return SimpleSeqRNN(*args, **kwargs)
    elif key.lower() == 'seqformer':
        return SeqTransformer(*args, **kwargs)
    #elif key.lower() == 'transformer':
    #    return Transformer(*args, **kwargs)