from mrc.Demo import RCData
from dotmap import DotMap
from mrc.model.OurModel import Model
import os
import pickle
from elasticsearch import Elasticsearch

es = Elasticsearch()
while True:
    print("请输入问题：")
    q = input()
    searched = es.search(index="news_index", body={
        'query': {
            'match': {
                'content': q,
            }
        }
    }, size=1)
    print(searched)
exit()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

dataName = 'search'

args = DotMap({
    "vocab_dir": './mrc/data/vocab/' + dataName + '/',
    "model_dir": './data/models/Our/' + dataName + '/',
    "max_p_num": 5,
    "max_p_len": 400,
    "max_q_len": 60,
    "max_ch_len": 20,
    "algo": "qanet"
})

print('Load vocab...')
with open(os.path.join(args.vocab_dir, dataName + 'OurVocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)


print('Restoring the model...')
rc_model = Model(vocab, args, demo=True)
rc_model.restore(args.model_dir, args.algo)

# passages = [
#     "",
# ]

passages = []
rc_data = RCData(args.max_p_num, args.max_p_len, args.max_q_len, args.max_ch_len, q, passages)
rc_data._convert_to_ids(vocab)

batch_data = rc_data.sample_to_batch(pad_id=vocab.get_word_id(vocab.pad_token),
                                     pad_char_id=vocab.get_char_id(vocab.pad_token))

result = rc_model.forward_one_sample(batch_data)
print(result[0]["pred_answers"])
