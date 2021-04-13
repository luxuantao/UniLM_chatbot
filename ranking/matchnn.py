import logging
import sys
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (AdamW, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup,
                          BertConfig,
                          BertForSequenceClassification
                          )
sys.path.append('..')
from config import is_cuda, max_sequence_length, root_path, bert_model
from ranking.data import DataPrecessForSentence


tqdm.pandas()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class BertModelTrain(nn.Module):
    """ The base model for training a matching network.
    """
    def __init__(self):
        super(BertModelTrain, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2)
        self.device = torch.device("cuda") if is_cuda else torch.device("cpu")
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        re = self.bert(input_ids=batch_seqs,
                                 attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments,
                                 labels=labels)
        loss, logits = re.loss, re.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class BertModelPredict(nn.Module):
    """ The base model for doing prediction using trained matching network.
    """
    def __init__(self):
        super(BertModelPredict, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2)
        self.device = torch.device("cuda") if is_cuda else torch.device("cpu")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments):
        re = self.bert(input_ids=batch_seqs,
                                attention_mask=batch_seq_masks,
                                token_type_ids=batch_seq_segments)
        logits = re.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities


class MatchingNN(object):
    """ The wrapper model for doing prediction using BertModelPredict.
    """
    def __init__(self,
                 model_path=os.path.join(root_path, 'model/ranking/best.pth.tar'),
                #  vocab_path=os.path.join(root_path, 'lib/bert/vocab.txt'),
                 data_path=os.path.join(root_path, 'data/ranking/test.tsv'),
                 is_cuda=is_cuda,
                 max_sequence_length=max_sequence_length):
        # self.vocab_path = vocab_path
        self.model_path = model_path
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.is_cuda = is_cuda
        self.device = torch.device('cuda') if self.is_cuda else torch.device('cpu')
        self.load_model()

    def load_model(self):
        self.model = BertModelPredict().to(self.device)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.dataPro = DataPrecessForSentence(self.bert_tokenizer,  # TODO
                                              self.data_path,
                                              self.max_sequence_length)

    def predict(self, q1, q2):
        result = [
            self.dataPro.trunate_and_pad(self.bert_tokenizer.tokenize(q1), self.bert_tokenizer.tokenize(q2))
        ]
        seqs = torch.Tensor([i[0] for i in result]).type(torch.long)
        seq_masks = torch.Tensor([i[1] for i in result]).type(torch.long)
        seq_segments = torch.Tensor([i[2] for i in result]).type(torch.long)

        if self.is_cuda:
            seqs = seqs.to(self.device)
            seq_masks = seq_masks.to(self.device)
            seq_segments = seq_segments.to(self.device)

        with torch.no_grad():
            res = self.model(seqs, seq_masks, seq_segments)[-1].cpu().detach().numpy()  # [[0.26710084 0.7328991 ]]
            label = res.argmax()
            score = res.tolist()[0][label]
            return label, score


if __name__ == '__main__':
    model = MatchingNN()
    print(model.predict('不对','你不好'))
