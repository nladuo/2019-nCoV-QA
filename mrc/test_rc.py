import os
import pickle
from dataset import BRCDataset
from vocab import Vocab
from rc_model import RCModel
import argparse
import jieba
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/preprocessed/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['./data/preprocessed/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/preprocessed/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='./data/preprocessed',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='./data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


class RCData:
    def __init__(self, max_p_num, max_p_len, max_q_len, q, passages):
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        question_tokens = jieba.lcut(q)
        self.sample = {
            "question_tokens": question_tokens,
            "passages": []
        }
        for passage in passages:
            self.sample["passages"].append({
                "passage_tokens": jieba.lcut(passage)
            })

    def _convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        question_token_ids = vocab.convert_to_ids(self.sample['question_tokens'])
        self.sample['question_token_ids'] = question_token_ids
        self.sample['question_length'] = len(question_token_ids)
        for passage in self.sample['passages']:
            passage_token_ids = vocab.convert_to_ids(passage['passage_tokens'])
            passage['passage_token_ids'] = passage_token_ids
            passage['passage_length'] = min(len(passage_token_ids), self.max_p_len)

    def sample_to_batch(self, pad_id):
        batch_data = {
            'raw_data': [self.sample],
            'question_token_ids': [],
            'question_length': [],
            'passage_token_ids': [],
            'passage_length': [],
            'start_id': [],
            'end_id': []
        }
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)

        for _ in batch_data['raw_data']:
            # fake span for some samples, only valid for testing
            batch_data['start_id'].append(0)
            batch_data['end_id'].append(0)

        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len


args = parse_args()

with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)


q = "2月10日，青海的新增病例是多少？"
passages = [
    "青海无新增确诊，累计确诊18例,2020年2月10日0-24时，青海省报告新型冠状病毒肺炎新增确诊病例0例，新增重症病例2例，新增死亡病例0例，新增出院病例0例。当日无新增确诊病例和疑似病例。截至2月10日24时，青海省累计报告新型冠状病毒肺炎确诊病例18例，其中：重症病例3例，死亡病例0例，出院病例3例。累计确诊病例中，西宁市15例，海北州3例，尚有15例确诊患者住院治疗。,青海卫健委",
    # "青海新增3例确诊，累计确诊18例,2020年2月11日0-24时"
]

rc_data = RCData(args.max_p_num, args.max_p_len, args.max_q_len, q, passages)

# exit()
rc_data._convert_to_ids(vocab)
print(rc_data.sample)

batch_data = rc_data.sample_to_batch(pad_id=vocab.get_id(vocab.pad_token))

# print(batch_data)

brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
rc_model = RCModel(vocab, args)
rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)

result1 = rc_model.forward_one_sample(batch_data)


passages = [
    # "青海无新增确诊，累计确诊18例,2020年2月10日0-24时，青海省报告新型冠状病毒肺炎新增确诊病例0例，新增重症病例2例，新增死亡病例0例，新增出院病例0例。当日无新增确诊病例和疑似病例。截至2月10日24时，青海省累计报告新型冠状病毒肺炎确诊病例18例，其中：重症病例3例，死亡病例0例，出院病例3例。累计确诊病例中，西宁市15例，海北州3例，尚有15例确诊患者住院治疗。,青海卫健委",
    "四川新增确诊12例，累计确诊417例,2月10日0-24时，我省新型冠状病毒肺炎新增确诊病例12例，新增治愈出院病例6例，无新增死亡病例。截至2月10日24时，我省累计报告新型冠状病毒肺炎确诊病例417例，"
]

rc_data = RCData(args.max_p_num, args.max_p_len, args.max_q_len, q, passages)
rc_data._convert_to_ids(vocab)
# print(rc_data.sample)

batch_data = rc_data.sample_to_batch(pad_id=vocab.get_id(vocab.pad_token))

result2 = rc_model.forward_one_sample(batch_data)


print(result1)
print(result2)


# feed_dict = {self.p: batch['passage_token_ids'],
#              self.q: batch['question_token_ids'],
#              self.p_length: batch['passage_length'],
#              self.q_length: batch['question_length'],
#              self.start_label: batch['start_id'],
#              self.end_label: batch['end_id'],
#              self.dropout_keep_prob: 1.0}
# start_probs, end_probs, loss = self.sess.run([self.start_probs,
#                                               self.end_probs, self.loss], feed_dict)


# rc_model.find_best_answer_for_passage()
# logger.info('Evaluating the model on dev set...')
# dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
#                                         pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
# dev_loss, dev_bleu_rouge = rc_model.evaluate(
#     dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')

