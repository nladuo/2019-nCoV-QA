import os
import jieba

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class RCData:
    def __init__(self, max_p_num, max_p_len, max_q_len, max_char_len, q, passages):
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_char_len = max_char_len

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
        self.sample['question_token_ids'] = vocab.convert_word_to_ids(self.sample['question_tokens'])
        self.sample["question_char_ids"] = vocab.convert_char_to_ids(self.sample['question_tokens'])
        for passage in self.sample['passages']:
            passage['passage_token_ids'] = vocab.convert_word_to_ids(passage['passage_tokens'])
            passage['passage_char_ids'] = vocab.convert_char_to_ids(passage['passage_tokens'])

    def sample_to_batch(self, pad_id, pad_char_id):

        batch_data = {
            'raw_data': [self.sample],
            'question_token_ids': [],
            'question_char_ids': [],
            'question_length': [],
            'passage_token_ids': [],
            'passage_length': [],
            'passage_char_ids': [],
            'start_id': [],
            'end_id': []
        }

        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_char_ids'].append(sample['question_char_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    passage_char_ids = sample['passages'][pidx]['passage_char_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                    batch_data['passage_char_ids'].append(passage_char_ids)
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['question_char_ids'].append([[]])
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
                    batch_data['passage_char_ids'].append([[]])

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id, pad_char_id)

        for _ in batch_data['raw_data']:
            # fake span for some samples, only valid for testing
            batch_data['start_id'].append(0)
            batch_data['end_id'].append(0)

        return batch_data

    def _dynamic_padding(self, batch_data, pad_id, pad_char_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_char_len = self.max_char_len
        pad_p_len = self.max_p_len  # min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = self.max_q_len  # min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        for index, char_list in enumerate(batch_data['passage_char_ids']):
            # print(batch_data['passage_char_ids'])
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['passage_char_ids'][index] = char_list
        batch_data['passage_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_p_len - len(ids)))[:pad_p_len]
                                          for ids in batch_data['passage_char_ids']]

        # print(np.array(batch_data['passage_char_ids']).shape, "==========")

        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        for index, char_list in enumerate(batch_data['question_char_ids']):
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['question_char_ids'][index] = char_list
        batch_data['question_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_q_len - len(ids)))[:pad_q_len]
                                           for ids in batch_data['question_char_ids']]

        return batch_data, pad_p_len, pad_q_len

