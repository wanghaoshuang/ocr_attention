"""Trainer for OCR Attention model."""
import os
import paddle.fluid as fluid
import attention_reader
import argparse
from load_model import load_param
import functools
import sys
from utility import add_arguments, print_arguments, to_lodtensor, get_attention_feeder_data
from attention_model import attention_train_net
from paddle.fluid import debuger

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',     int,   1,     "Minibatch size.")
add_arg('pass_num',       int,   2,     "# of training epochs.")
add_arg('log_period',     int,   1000,   "Log period.")
add_arg('learning_rate',  float, 0.0, "Learning rate.")
add_arg('l2',             float, 0.0004, "L2 regularizer.")
add_arg('max_clip',       float, 10.0,   "Max clip threshold.")
add_arg('min_clip',       float, -10.0,  "Min clip threshold.")
add_arg('rnn_hidden_size',int,   128,    "Hidden size of rnn layers.")
add_arg('device',         int,   0,      "Device id.'-1' means running on CPU"
                                         "while '0' means GPU-0.")
add_arg('beam_size',     int,   5,     "Beam size.")

# yapf: disable

def load_parameter(place):
    params = load_param('./attention_name.map', './attention_data/pass-00002/')
    for name in params:
        t = fluid.global_scope().find_var(name).get_tensor()
        t.set(params[name], place)
        #if name == 'gru_unit_0.b_0':
        #    print "gru_unit_0.b_0:\n %s" % params[name]


def train(args, data_reader=attention_reader):
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
    label_in = fluid.layers.data(name='label_in', shape=[1], dtype='int32', lod_level=1)
    label_out = fluid.layers.data(name='label_out', shape=[1], dtype='int32', lod_level=1)
    #sum_cost, error_evaluator, beam_gen = attention_train_net(images, label_in, label_out, args, num_classes)
    sum_cost, error_evaluator = attention_train_net(images, label_in, label_out, args, num_classes)
    #####
    #for param in fluid.default_main_program().global_block().all_parameters():
    #    print "%s=%s" % (param.name, param.shape)
    #####
    #vars = fluid.default_main_program().list_vars()
    #for var in vars:
    #    print "%s %s" % (var.name, var.shape)
    debuger.draw_block_graphviz(fluid.default_main_program().global_block(), path="./map.dot")
    # data reader
    train_reader = data_reader.train(args.batch_size)
    test_reader = data_reader.test()
    # prepare environment
    place = fluid.CPUPlace()
    if args.device >= 0:
        place = fluid.CUDAPlace(args.device)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    load_parameter(place)

    inference_program = fluid.io.get_inference_program(error_evaluator)

    #for pass_id in range(args.pass_num):
    for pass_id in range(1):
        error_evaluator.reset(exe)
        batch_id = 1
        total_loss = 0.0
        total_seq_error = 0.0
        # train a pass
        #for data in train_reader():
        for data in test_reader():
            batch_loss, _, batch_seq_error = exe.run(
                fluid.default_main_program(),
                feed=get_attention_feeder_data(data, place),
                fetch_list=[sum_cost] + error_evaluator.metrics)
            total_loss += batch_loss[0]
            total_seq_error += batch_seq_error[0]
            if batch_id % 10 == 1:
                print '.',
            if batch_id % args.log_period == 1:
                print "\nPass[%d]-batch[%d]; Avg Attention loss: %s; Avg seq error: %s." % (
                    pass_id, batch_id, total_loss / (batch_id * args.batch_size), total_seq_error / (batch_id * args.batch_size))
            batch_id += 1
            ####
     #       model_path = "./new_model_attention/pass_" + str(pass_id)
     #       os.mkdir(model_path)
     #       fluid.io.save_params(exe, model_path)
     #       return

        #####
#        error_evaluator.reset(exe)
#        for data in test_reader():
#            exe.run(inference_program, feed=get_attention_feeder_data(data, place))
#        _, test_seq_error = error_evaluator.eval(exe)
#        print "\nEnd pass[%d]; Test seq error: %s.\n" % (
#            pass_id, str(test_seq_error[0]))

      #  model_path = "./new_model_attention/" + str(pass_id)
      #  if not os.path.exists(model_path):
      #      os.mkdir(model_path)
      #  fluid.io.save_inference_model(model_path,
      #                               ["pixel"],
      #                               [beam_gen],
      #                               exe,
      #                               main_program=None,
      #                               model_filename='model',
      #                               params_filename='params')

def main():
    args = parser.parse_args()
    print_arguments(args)
    train(args, data_reader=attention_reader)

if __name__ == "__main__":
    main()
