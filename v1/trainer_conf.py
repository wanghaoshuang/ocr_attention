#edit-mode: -*- python -*-
#coding:gbk
import os
import sys
import math
import paddle.trainer.recurrent_units as recurrent
from paddle.trainer_config_helpers import *

is_generating = get_config_arg("generating", bool, False)
gen_trans_file = get_config_arg("gen_trans_file", str, None)
trg_lang_dict = "dict_line"

num_classes = 95 + 2
sos = 0
eos = 1

beam_size = get_config_arg("beam_size", int, 5)

clipping_thd = 10

default_gradient_clipping_threshold(10)

decoder_size = 128
word_vector_dim = 128
max_length = 100

if is_generating:
    ASYNC=False
else:
    ASYNC=True

TrainData(PyData(
    #files = "/home/dangqingqing/suzhizhong/lixuan/eng_train/20conv_att_108w/list/108w_img_list",
    files = "/home/vis/lixuan/paddle_train/eng_att_fluid_duiqi/train_list/test_sample.list",
    load_data_module="pyImageDataProvider_attention_batch",
    load_data_object="SimpleDataProvider",
    async_load_data=ASYNC
))

if is_generating:
    TestData(PyData(
        #files = "/home/dangqingqing/suzhizhong/lixuan/eng_train/20conv_att_108w/list/dev_img_list",
        files = "/home/vis/lixuan/paddle_train/eng_att_fluid_duiqi/train_list/test_sample.list",
        load_data_module="pyImageDataProvider_gen_attention_batch",
        load_data_object="SimpleDataProvider_test"
    ))
else:
    TestData(PyData(
        files = "/home/vis/lixuan/paddle_train/eng_att_fluid_duiqi/train_list/test_sample.list",
        #files = "/home/dangqingqing/suzhizhong/lixuan/eng_train/20conv_att_108w/list/dev_img_list",
        load_data_module="pyImageDataProvider_test_attention_batch",
        load_data_object="SimpleDataProvider_test"
    ))

settings(
    batch_size=1,
    learning_method = AdaDeltaOptimizer(rho=0.9),
    learning_rate = 0,
    #regularization = L2Regularization(0.0),
    #learning_rate = (1.0e-3) / 32,
    #learning_method = MomentumOptimizer(0.),
    #learning_rate_decay_a=0.1,
    #learning_rate_decay_b=3588000 * 10,
    #learning_rate_schedule="discexp",
)

def conv_bn_pool(input, group, out_ch, in_ch=None, pool=True):
    tmp = input
    for i in xrange(group):
        tmp = img_conv_layer(input=tmp,
                             filter_size=3,
                             num_channels=in_ch[i] if in_ch is not None else None,
                             num_filters=out_ch[i],
                             stride=1,
                             padding=1,
                             act=LinearActivation(),
                             bias_attr=False)
        tmp = batch_norm_layer(input=tmp, act=ReluActivation())
    if pool:
        tmp = img_pool_layer(input=tmp, stride=2, pool_size=2, pool_type=MaxPooling())
    else:
        tmp = tmp
    return tmp

def ocr_convs(input_image, num, with_bn):
  assert(num % 4 == 0)
  tmp = input_image
  tmp = conv_bn_pool(tmp, 2, [16, 16], [1, 16])
  tmp = conv_bn_pool(tmp, 2, [32, 32])
  tmp = conv_bn_pool(tmp, 2, [64, 64])
  tmp = conv_bn_pool(tmp, 2, [128, 128], pool=False)
  return tmp

# data layers
img = data_layer(name = "input", size = 2304)

# encoder part
conv_features = ocr_convs(img, 20, True)

#gradient_printer_evaluator(conv_features)
sliced_feature = block_expand_layer(input = conv_features,
                                    num_channels = 128,
                                    stride_x = 1,
                                    stride_y = 1,
                                    block_x = 1,
                                    block_y = 6,)

para_attr = ParamAttr(initial_mean=0., initial_std=0.02)
bias_attr = ParamAttr(initial_mean=0.0, initial_std=0.02, learning_rate=2.)
gru_forward = simple_gru2(input = sliced_feature, size = 128, act = ReluActivation(),
                          mixed_param_attr = para_attr, mixed_bias_attr=False,
                          gru_param_attr = para_attr, gru_bias_attr = bias_attr)
gru_backward = simple_gru2(input = sliced_feature, size = 128, reverse = True, act = ReluActivation(),
                           mixed_param_attr = para_attr, mixed_bias_attr=False,
                           gru_param_attr = para_attr, gru_bias_attr = bias_attr)

encoded_vector = concat_layer(input = [gru_forward, gru_backward])
#eval_4 = value_printer_evaluator(input = encoded_vector)

with mixed_layer(size=decoder_size) as encoded_proj:
  encoded_proj += full_matrix_projection(input=encoded_vector)
#eval_5 = value_printer_evaluator(input = encoded_proj)

# decoder part
backward_first = first_seq(input=gru_backward)

with mixed_layer(size=decoder_size, act=ReluActivation()) as decoder_boot:
  decoder_boot += full_matrix_projection(input=backward_first)
#eval_6 = value_printer_evaluator(input = decoder_boot)

#eval1 = value_printer_evaluator(input=decoder_boot)

def gru_decoder_with_attention(enc_vec, enc_proj, current_word):
  decoder_mem = memory(
    name='gru_decoder', size=decoder_size, boot_layer=decoder_boot)
  context = simple_attention(
    #name="attention",
    encoded_sequence=enc_vec,
    encoded_proj=enc_proj,
    decoder_state=decoder_mem, )
  with mixed_layer(size=decoder_size * 3) as decoder_inputs:
    decoder_inputs += full_matrix_projection(input=context)
    decoder_inputs += full_matrix_projection(input=current_word)

  gru_step = gru_step_layer(
    name='gru_decoder',
    input=decoder_inputs,
    output_mem=decoder_mem,
    size=decoder_size)
  gradient_printer_evaluator(input=gru_step)
  with mixed_layer(
    #name='prob_matrix',
    size=num_classes, bias_attr=True,
    act=LinearActivation()) as out:
    out += full_matrix_projection(input=gru_step)

  gradient_printer_evaluator(input=out)
  return out

decoder_group_name = "decoder_group"
group_inputs = [
      StaticInput(input=encoded_vector, is_seq=True),
      StaticInput(input=encoded_proj, is_seq=True)
]

# training
if not is_generating:
  label_in = data_layer(name = "label_in", size = num_classes)
  label_out = data_layer(name = "label_out", size = num_classes)
  trg_embedding = embedding_layer(
    input=label_in,
    size=word_vector_dim,
    param_attr=ParamAttr(name='_target_language_embedding'))

  group_inputs.append(trg_embedding)

  decoder = recurrent_group(
    name=decoder_group_name,
    step=gru_decoder_with_attention, input=group_inputs)
  cost = classification_cost(input=decoder, label=label_out)

#  gradient_printer_evaluator(cost)
#  value_printer_evaluator(input=cost)
#  maxid = maxid_layer(input=decoder)
  #eval1 = value_printer_evaluator(input=maxid)
#  eval = seq_error_evaluator(input=maxid, label=label_out, uncare=[eos, sos])
  outputs(cost)

# generating
else:
  pass
  trg_embedding = GeneratedInput(
      size=num_classes,
      embedding_name='_target_language_embedding',
      embedding_size=word_vector_dim)
  group_inputs.append(trg_embedding)

  beam_gen = beam_search(
      name=decoder_group_name,
      step=gru_decoder_with_attention,
      input=group_inputs,
      bos_id=sos,
      eos_id=eos,
      beam_size=beam_size,
      max_length=max_length)

  seqtext_printer_evaluator(
      input=beam_gen,
      dict_file=trg_lang_dict,
      result_file=gen_trans_file)

  outputs(beam_gen)
  #Outputs("attention_softmax", "__beam_search_predict__", "prob_matrix@decoder_group")

