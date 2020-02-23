from mrc.Demo import RCData
from dotmap import DotMap
from mrc.model.OurModel import Model
import os
import pickle
from elasticsearch import Elasticsearch
import time
import datetime
import warnings
warnings.filterwarnings("ignore")


es = Elasticsearch()


def reconstruct_q(q):
    today = datetime.date.today()
    # 昨天时间
    yesterday = today - datetime.timedelta(days=1)
    yesterday_str = f"{yesterday.year}年{yesterday.month}月{yesterday.day}日"
    nowadays = [
        "最新",
        "今日",
        "今天"
    ]
    for d in nowadays:
        if d in q:
            q = q.replace(d, yesterday_str+"0时至24时")
            return q, today.strftime("%Y-%m-%d")
    return q, ""


# exit()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

dataName = 'search'

args = DotMap({
    "vocab_dir": './mrc/data/vocab/' + dataName + '/',
    "model_dir": './mrc/data/models/Our/' + dataName + '/',
    "summary_dir": './mrc/data/summary/Our/' + dataName + '/',
    "log_path": './mrc/data/summary/Our/' + dataName + '/log.txt',
    "max_p_num": 5,
    "max_p_len": 400,
    "max_q_len": 60,
    "max_ch_len": 20,
    "max_a_len": 200,
    "use_position_attn": True,
    "char_embed_size": 32,
    "word_embed_size": 150,
    "algo": "qanet",
    "dropout": 0,
    "hidden_size": 64,
    "head_size": 1,
    "loss_type": "cross_entropy",
    "fix_pretrained_vector": True,
    "optim": "adam",
    "learning_rate": 0.00005,
    "weight_decay": 1e-5,
    "decay": None,
    "l2_norm": 3e-7,
    'clip_weight': True,
    "max_norm_grad": 5.0,
    "batch_size": 16,
    "epochs": 10
})

print('Load vocab...')
with open(os.path.join(args.vocab_dir, dataName + 'OurVocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)


print('Restoring the model...')
rc_model = Model(vocab, args, demo=True)
rc_model.restore(args.model_dir, args.algo)

while True:
    print("\n请输入问题：")
    q = input()
    q, a = reconstruct_q(q)
    # print(q)
    searched = es.search(index="news_index", body={
        'query': {
            'match': {
                'content': q + " " + a,
            }
        }
    }, size=1)
    content = searched["hits"]["hits"][0]["_source"]["content"]
    passages = [content]

    rc_data = RCData(args.max_p_num, args.max_p_len, args.max_q_len, args.max_ch_len, q, passages)
    rc_data._convert_to_ids(vocab)

    batch_data = rc_data.sample_to_batch(pad_id=vocab.get_word_id(vocab.pad_token),
                                         pad_char_id=vocab.get_char_id(vocab.pad_token))

    result = rc_model.forward_one_sample(batch_data)
    print(result[0]["pred_answers"][0])
    print("答案得分：", result[0]["pred_score"][0])
