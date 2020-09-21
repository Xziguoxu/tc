import tensorflow as tf
import numpy as np
import random
import time
from model_seq2seq_contrib import Seq2seq
# from model_seq2seq import Seq2seq

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True


class Config(object):
    embedding_dim = 100
    hidden_dim = 50
    batch_size = 128
    learning_rate = 0.005
    source_vocab_size = None
    target_vocab_size = None

def str2int(lt):
    re=[]
    for x in lt:
        re.append(int(x))
    return re

def load_data(path):
     num2en = {"TELNET": 0, "FTP": 1, "HTTP": 2, "DNS": 3, "lime": 4, "localForwarding": 5, "remoteForwarding": 6, "scp": 7, "sftp": 8, "x11": 9, "shell": 10}
     docs_source = []
     docs_target = []
     label_file = []
     with open("NIMS_train.arff", "r") as f:
         label_file = f.readlines()
     for line in label_file:
         tmp = line.split(',')
         print(tmp[:-1])
         print(tmp[-1])
         break
     for line in label_file:
         tmp = line.split(',')
         # docs_source.append(tmp[:-1])
         docs_source.append(str2int(tmp[:-1]))
         docs_target.append([ tmp[-1].strip("\n") ])
     return docs_source, docs_target

def load_test_data(path):
     num2en = {"TELNET": 0, "FTP": 1, "HTTP": 2, "DNS": 3, "lime": 4, "localForwarding": 5, "remoteForwarding": 6, "scp": 7, "sftp": 8, "x11": 9, "shell": 10}
     docs_source = []
     docs_target = []
     label_file = []
     with open("NIMS_test.arff", "r") as f:
         label_file = f.readlines()
     for line in label_file:
         tmp = line.split(',')
         print(tmp[:-1])
         print(tmp[-1])
         break
     for line in label_file:
         tmp = line.split(',')
         # docs_source.append(tmp[:-1])
         docs_source.append(str2int(tmp[:-1]))
         docs_target.append([ tmp[-1].strip("\n") ])
     return docs_source, docs_target

def make_vocab(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    for doc in docs:
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
    return w2i, i2w


def make_target_vocab(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    for doc in docs:
        for w in doc:
          if w not in w2i:
            i2w[len(w2i)] = w
            w2i[w] = len(w2i)
    return w2i, i2w


def doc_to_seq(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    seqs = []
    for doc in docs:
        seq = []
        for w in doc:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            seq.append(w2i[w])
        seqs.append(seq)
    return seqs, w2i, i2w


def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size):
    ps = []
    while len(ps) < batch_size:
        ps.append(random.randint(0, len(docs_source)-1))

    source_batch = []
    target_batch = []

    #print("docs target:", docs_target[:10])
    source_lens = [len(docs_source[p]) for p in ps]
    #target_lens = [len(docs_target[p])+1 for p in ps]
    target_lens = [ len(docs_target[p]) + 1 for p in ps]
    #print("target len:", target_lens[:10])

    max_source_len = max(source_lens)
    max_target_len = max(target_lens)
    #print("max source len:", max_source_len)
    #print("max target len:", max_target_len)
    for p in ps:
        source_seq = [w2i_source[w] for w in docs_source[p]] + \
            [w2i_source["_PAD"]]*(max_source_len-len(docs_source[p]))
        #target_seq = [w2i_target[docs_target[p]]] + [w2i_target["_EOS"]] + [w2i_target["_PAD"]]*(max_target_len-1)
        #target_seq = [w2i_target[docs_target[p]]] + [w2i_target["_EOS"]]
        #target_seq = [w2i_target[w] for w in docs_target[p]] +  [w2i_target["_EOS"]] + [w2i_target["_PAD"]]*(max_target_len-1-len(docs_target[p]))
        target_seq = [w2i_target[w] for w in docs_target[p]] +  [w2i_target["_EOS"]]
        source_batch.append(source_seq)
        target_batch.append(target_seq)

    return source_batch, source_lens, target_batch, target_lens


if __name__ == "__main__":

    print("(1)load data......")
    docs_source, docs_target = load_data("")
    print("source doc[:10]", docs_source[:2])
    print("target doc[:10]", docs_target[:2])
    w2i_source, i2w_source = make_vocab(docs_source)
    #w2i_target, i2w_target = make_vocab(docs_target)
    w2i_target, i2w_target = make_target_vocab(docs_target)
    print("w2i_target[0]", w2i_target.keys())
    print("(2) build model......")
    config = Config()
    config.source_vocab_size = len(w2i_source)
    config.target_vocab_size = len(w2i_target)
    print("source size", config.source_vocab_size)
    print("target size", config.target_vocab_size)
    model = Seq2seq(config=config, w2i_target=w2i_target,
                    useTeacherForcing=True, useAttention=True, useBeamSearch=1)

    print("(3) run model......")
    batches = 3000
    print_every = 100

    saver = tf.train.Saver()
    with tf.Session(config=tf_config) as sess:
        tf.summary.FileWriter('graph', sess.graph)
        sess.run(tf.global_variables_initializer())

        losses = []
        total_loss = 0
        for batch in range(batches):
            source_batch, source_lens, target_batch, target_lens = get_batch(
                docs_source, w2i_source, docs_target, w2i_target, config.batch_size)

            feed_dict = {
                model.seq_inputs: source_batch,
                model.seq_inputs_length: source_lens,
                model.seq_targets: target_batch,
                model.seq_targets_length: target_lens
            }

            loss, _ = sess.run([model.loss, model.train_op], feed_dict)
            total_loss += loss

            #print("target batch: ", target_batch[:10])

            if batch % print_every == 0 and batch > 0:
                print_loss = total_loss if batch == 0 else total_loss / print_every
                losses.append(print_loss)
                total_loss = 0
                print("-----------------------------")
                print("batch:", batch, "/", batches)
                print("time:", time.strftime(
                    '%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print("loss:", print_loss)

                print("samples:\n")
                predict_batch = sess.run(model.out, feed_dict)
                count = 0
                for i in range(64):
                    if (predict_batch[i]==target_batch[i]).all():
                        count = count + 1
                print("acc: ", count / 64.0)
                for i in range(3):
                    print("in:", [i2w_source[num]
                                  for num in source_batch[i] if i2w_source[num] != "_PAD"])
                    print("out:", [i2w_target[num]
                                   for num in predict_batch[i] if i2w_target[num] != "_PAD"])
                    print("tar:", [i2w_target[num]
                                   for num in target_batch[i] if i2w_target[num] != "_PAD"])
                    print("")

        print(losses)
        print(saver.save(sess, "checkpoint/model.ckpt"))
