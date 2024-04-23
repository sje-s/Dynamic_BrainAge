from models.bilstm import BiLSTM
from models.gru import BiGRU
from models.transformer import MyTransformer
from models.sequence_modeling.seq_rnn import SeqRNN
from models.sequence_modeling.simple_rnn import SimpleSeqRNN
from models.sequence_modeling.transformer.transformer import SeqTransformer


def get_model(key, *args, **kwargs):
    if key.lower() == "bilstm":
        return BiLSTM(*args, **kwargs)
    elif key.lower() == "bigru":
        return BiGRU(*args, **kwargs)
    elif key.lower() == "seqrnn":
        return SeqRNN(*args, **kwargs)
    elif key.lower() == 'simpleseqrnn':
        return SimpleSeqRNN(*args, **kwargs)
    elif key.lower() == 'seqformer':
        return SeqTransformer(*args, **kwargs)
    elif key.lower() == 'transformer':
        return Transformer(*args, **kwargs)