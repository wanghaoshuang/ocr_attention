import paddle.fluid as fluid

decoder_size = 128
word_vector_dim = 128
max_length = 100
sos = 0
eos = 1

def conv_bn_pool(input,
                 group,
                 out_ch,
                 act="relu",
                 param=None,
                 bias=None,
                 param_0=None,
                 is_test=False,
                 pool = True):
    tmp = input
    for i in xrange(group):
        tmp = fluid.layers.conv2d(
            input=tmp,
            num_filters=out_ch[i],
            filter_size=3,
            padding=1,
            bias_attr=False,
            param_attr=param if param_0 is None else param_0,
            act=None,  # LinearActivation
            use_cudnn=True)
        tmp = fluid.layers.batch_norm(
            input=tmp,
            act=act,
            param_attr=param,
            bias_attr=bias,
            is_test=is_test)
    if pool == True:
        tmp = fluid.layers.pool2d(
            input=tmp, pool_size=2, pool_type='max', pool_stride=2, use_cudnn=True, ceil_mode=True)

    return tmp


def ocr_convs(input,
              num,
              with_bn,
              regularizer=None,
              gradient_clip=None,
              is_test=False):
    assert (num % 4 == 0)

    b = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.0))
    w0 = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.0005))
    w1 = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.01))
    tmp = input
    tmp = conv_bn_pool(
        tmp, 2, [16, 16], param=w1, bias=b, param_0=w0, is_test=is_test)
    tmp = conv_bn_pool(tmp, 2, [32, 32], param=w1, bias=b, is_test=is_test)
    tmp = conv_bn_pool(tmp, 2, [64, 64], param=w1, bias=b, is_test=is_test)
    tmp = conv_bn_pool(tmp, 2, [128, 128], param=w1, bias=b, is_test=is_test, pool=False)
    return tmp


def encoder_net(images,
                rnn_hidden_size,
                regularizer=None,
                gradient_clip=None,
                is_test=False):

    conv_features = ocr_convs(
        images,
        8,
        True,
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        is_test=is_test)

    sliced_feature = fluid.layers.im2sequence(
        input=conv_features,
        stride=[1, 1],
        filter_size=[conv_features.shape[2], 1])

    para_attr = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.02))
    bias_attr = fluid.ParamAttr(
        regularizer=regularizer,
        gradient_clip=gradient_clip,
        initializer=fluid.initializer.Normal(0.0, 0.02),
        learning_rate=2.0)

    fc_1 = fluid.layers.fc(input=sliced_feature,
                           size=rnn_hidden_size * 3,
                           param_attr=para_attr,
                           bias_attr=False)
    fc_2 = fluid.layers.fc(input=sliced_feature,
                           size=rnn_hidden_size * 3,
                           param_attr=para_attr,
                           bias_attr=False)

    gru_forward = fluid.layers.dynamic_gru(
        input=fc_1,
        size=rnn_hidden_size,
        param_attr=para_attr,
        bias_attr=bias_attr,
        candidate_activation='relu')
    gru_backward = fluid.layers.dynamic_gru(
        input=fc_2,
        size=rnn_hidden_size,
        is_reverse=True,
        param_attr=para_attr,
        bias_attr=bias_attr,
        candidate_activation='relu')

    encoded_vector = fluid.layers.concat(input=[gru_forward, gru_backward],
                                         axis=1)
    encoded_proj = fluid.layers.fc(input=encoded_vector,
                                   size=decoder_size,
                                   bias_attr=False)

    return gru_backward, encoded_vector, encoded_proj


def attention_train_net(images, label_in, label_out, args, num_classes):
    regularizer = fluid.regularizer.L2Decay(args.l2)
    gradient_clip = None
    gru_backward, encoded_vector, encoded_proj = encoder_net(
        images,
        args.rnn_hidden_size,
        regularizer=regularizer,
        gradient_clip=gradient_clip)

    backward_first = fluid.layers.sequence_pool(input=gru_backward,
                                                pool_type='first')
    decoder_boot  = fluid.layers.fc(input=backward_first,
                                    size=decoder_size,
                                    bias_attr=False,
                                    act="relu")

    def gru_decoder_with_attention(target_embedding, encoder_vec,
                    encoder_proj, decoder_boot, decoder_size):
        def simple_attention(encoder_vec, encoder_proj, decoder_state):
            decoder_state_proj = fluid.layers.fc(input=decoder_state,
                                                 size=decoder_size,
                                                 bias_attr=False)
            decoder_state_expand = fluid.layers.sequence_expand(
                x=decoder_state_proj, y=encoder_proj)
            concated = encoder_proj + decoder_state_expand
            concated = fluid.layers.tanh(x=concated)
            attention_weights = fluid.layers.fc(input=concated,
                                                size=1,
                                                act=None,
                                                bias_attr=False)
            attention_weights = fluid.layers.sequence_softmax(
                input=attention_weights)
            weigths_reshape = fluid.layers.reshape(
                x=attention_weights, shape=[-1])
            scaled = fluid.layers.elementwise_mul(
                x=encoder_vec, y=weigths_reshape, axis=0)
            context = fluid.layers.sequence_pool(input=scaled, pool_type='sum')
            return context


        rnn = fluid.layers.DynamicRNN()

        with rnn.block():
            current_word = rnn.step_input(target_embedding)
            encoder_vec = rnn.static_input(encoder_vec)
            encoder_proj = rnn.static_input(encoder_proj)
            hidden_mem = rnn.memory(init=decoder_boot, need_reorder=True)
            context = simple_attention(encoder_vec, encoder_proj, hidden_mem)
            fc_1 = fluid.layers.fc(input=context,
                    size=decoder_size * 3, bias_attr=False)
            fc_2 = fluid.layers.fc(input=current_word,
                    size=decoder_size * 3, bias_attr=False)
            decoder_inputs = fc_1 + fc_2
            h, _, _ = fluid.layers.gru_unit(
                        input=decoder_inputs,
                        hidden = hidden_mem,
                        size=decoder_size * 3)
            rnn.update_memory(hidden_mem, h)
            h = fluid.layers.Print(input=h, print_phase="backward")
            out = fluid.layers.fc(input=h,
                                  size=num_classes + 2,
                                  bias_attr=True)
#                                  act='softmax')
            out = fluid.layers.Print(out, print_phase="backward")
            rnn.output(out)
        return rnn()

#training
    label_in = fluid.layers.cast(x=label_in, dtype='int64')
    trg_embedding = fluid.layers.embedding(
        input=label_in,
        size=[num_classes + 2, word_vector_dim],
        dtype='float32')
    prediction = gru_decoder_with_attention(trg_embedding, encoded_vector,
            encoded_proj, decoder_boot,
            decoder_size)


    label_out = fluid.layers.cast(x=label_out, dtype='int64')

    cost = fluid.layers.cross_entropy(input=prediction, label=label_out)
    sum_cost = fluid.layers.reduce_mean(cost)
    optimizer = fluid.optimizer.Adadelta(
                    learning_rate=args.learning_rate,
                    epsilon=1.0e-6,
                    rho=0.9)
    optimizer.minimize(sum_cost)

    _,maxid = fluid.layers.topk(input=prediction, k=1)
    casted_label = fluid.layers.cast(x=label_out, dtype='int64')
    error_evaluator = fluid.evaluator.EditDistance(
                        input=maxid,
                        label=casted_label,
                        ignored_tokens = [0, 1])

#inference
 #   trg_embedding = fluid.layers.GeneratedInput(
 #       size=num_classes + 2,
 #       embedding_size=word_vector_dim)
 #   group_inputs_infer.append(trg_embedding)##xinkaiyige

 #   beam_gen = fluid.layers.beam_search(
 #       step=gru_decoder_with_attention,
 #       input=group_inputs_infer,
 #       bos_id=sos,
 #       eos_id=eos,
 #       beam_size=args.beam_size,
 #       max_length=max_length)

 #   beam_gen = fluid.layers.cast(x=beam_gen, dtype='float32')

 #   return sum_cost, error_evaluator, beam_gen
    return sum_cost, error_evaluator

