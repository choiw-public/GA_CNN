from itertools import accumulate
from tensorflow.python.framework import meta_graph
import random
import tensorflow as tf
import numpy as np
import time
import os
import re


class CommonProperty:
    def __init__(self, model_depth_range, conv_size_range, conv_depth_range, conv_stride_range, num_population, selection_rule, mutation_num, best_parent_num,
                 num_classes):
        self.model_depth_range = model_depth_range
        self.conv_size_range = conv_size_range
        self.conv_depth_range = conv_depth_range
        self.conv_stride_range = conv_stride_range
        self.num_population = num_population
        self.selection_rule = selection_rule
        self.mutation_num = mutation_num
        self.best_parent_num = best_parent_num
        self.conv_size_bit_num = len(format(conv_size_range[1], "b"))
        self.conv_depth_bit_num = len(format(conv_depth_range[1], "b"))
        self.conv_stride_bit_num = len(format(conv_stride_range[1], "b"))
        self.per_conv_bit_num = self.conv_size_bit_num + self.conv_depth_bit_num + self.conv_stride_bit_num
        self.max_model_length = model_depth_range[1]
        self.model_length_bit_num = self.max_model_length * self.per_conv_bit_num
        self.num_classes = num_classes


def get_shape(tensor):
    _static_shape = tensor.get_shape().as_list()
    _dynamic_shape = tf.unstack(tf.shape(tensor))
    _dims = [s[1] if s[0] is None else s[0] for s in zip(_static_shape, _dynamic_shape)]
    return _dims


def list_getter(dir_name, extension, must_include=None):
    def sort_nicely(a_list):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        a_list.sort(key=alphanum_key)

    file_list = []
    if dir_name:
        for path, subdirs, files in os.walk(dir_name):
            for name in files:
                if name.lower().endswith((extension)):
                    if must_include:
                        if must_include in name:
                            file_list.append(os.path.join(path, name))
                    else:
                        file_list.append(os.path.join(path, name))
        sort_nicely(file_list)
    if not file_list:
        raise ValueError("list is empty")
    return file_list


def adopt_datapipeline_base(data_graph):
    model_graph = tf.get_default_graph()
    data_graph_meta = tf.train.export_meta_graph(graph=data_graph)
    meta_graph.import_scoped_meta_graph(data_graph_meta, import_scope="data_graph")

    train_data = {"image": model_graph.get_tensor_by_name("data_graph/IteratorGetNext:1"),
                  "gt": model_graph.get_tensor_by_name("data_graph/IteratorGetNext:0"),
                  "init": model_graph.get_operation_by_name("data_graph/MakeIterator")}

    test_data = {"image": model_graph.get_tensor_by_name("data_graph/IteratorGetNext_1:1"),
                 "gt": model_graph.get_tensor_by_name("data_graph/IteratorGetNext_1:0"),
                 "init": model_graph.get_operation_by_name("data_graph/MakeIterator_1")}
    return train_data, test_data


def train_and_test(sess, max_epoch, train_data, test_data, model, lr, current_epoch):
    with tf.control_dependencies(tf.get_collection('update_ops')):
        train_op = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(model.loss)
    sess.run(tf.global_variables_initializer())
    print("\n   ===============================================================")
    print("       Name:%s, Model length: %02d, Parameters: %10d" % (model.name, model.model_length, model.total_parameters))
    print("   ===============================================================")
    print("      * Convolution size  : |%s|" % ("| ".join(["%3d" % entry for entry in model.conv_size])))
    print("      * Convolution depth : |%s|" % ("| ".join(["%3d" % entry for entry in model.conv_depth])))
    print("      * Convolution stride: |%s|" % ("| ".join(["%3d" % entry for entry in model.conv_stride])))
    print("   ---------------------------------------------------------------")
    print("      * Accuracy history:", end="")
    tic = time.time()
    for epoch in range(max_epoch):
        sess.run(train_data["init"])
        while True:
            try:  # repeat train step
                # _, loss, learning_rate = sess.run([train_op, model.loss, lr], {current_epoch: epoch, is_train: True}) # for monitoring purpose
                sess.run([train_op], {current_epoch: epoch})
            except tf.errors.OutOfRangeError:  # if train step reached to an epoch, start test with test dataset
                test_num = 0
                tf_pred = tf.argmax(model.build(test_data["image"], False), 1)  # prediction tensor with test data
                sess.run(test_data["init"])
                correct = 0
                while True:
                    try:
                        pred, gt = sess.run([tf_pred, test_data["gt"]])
                        test_num += len(pred)
                        correct += np.sum(pred == gt)
                    except tf.errors.OutOfRangeError:
                        accuracy = float(correct) / float(test_num)
                        break
                break
        if epoch % 5 == 0:
            print("\n          %.2f >>> " % accuracy, end="")
        else:
            print("%.2f >>> " % accuracy, end="")
    print("\n      * Duration: %.3f minutes \n" % ((time.time() - tic) / 60.0))
    return accuracy


def fitness_fn(total_population):
    max_param = total_population["parameter"].max()
    fitness = total_population["accuracy"] + (1.0 - (total_population["parameter"] / max_param))
    total_population["fitness"] = fitness
    return total_population.sort_values("fitness", ascending=False).reset_index(drop=True)  # sort in descending order


def add_or_drop_child(child_actual_bit, offspring_pool, total_gene, common):
    conv_size, conv_depth, conv_stride = [], [], []
    actual_length = int(len(child_actual_bit) / common.per_conv_bit_num)
    for idx in range(actual_length):
        conv_spec = child_actual_bit[common.per_conv_bit_num * idx: common.per_conv_bit_num * (idx + 1)]
        conv_size.append(int(conv_spec[0:common.conv_size_bit_num], 2))
        conv_depth.append(int(conv_spec[common.conv_size_bit_num: common.conv_size_bit_num + common.conv_depth_bit_num], 2))
        conv_stride.append(int(conv_spec[common.conv_size_bit_num + common.conv_depth_bit_num:
                                         common.conv_size_bit_num + common.conv_depth_bit_num + common.conv_stride_bit_num], 2))
    child_model_bit = child_actual_bit.zfill(common.model_length_bit_num)
    if 0 not in conv_size + conv_depth + conv_stride and child_model_bit not in total_gene:
        offspring_pool.append({"conv_size": conv_size,
                               "conv_depth": conv_depth,
                               "conv_stride": conv_stride,
                               "model_bit": child_model_bit})
    return offspring_pool


def crossover(total_population, crossover_num, common):
    def select_parent(total_population, cumulative_probability, index_table):
        random_prob = np.random.uniform()
        for idx, prob in enumerate(cumulative_probability):
            if random_prob <= prob:
                index_start = index_table[idx]
                index_end = index_table[idx + 1]
                return total_population.iloc[index_start:index_end, :].sample()

    def get_bit_window(actual_bit, bit_window_size):
        if len(actual_bit) == bit_window_size:
            return actual_bit, [0, len(actual_bit)]
        else:
            start = np.random.randint(low=0, high=len(actual_bit) - bit_window_size)
            end = start + bit_window_size
            return actual_bit[start:end], [start, end]

    total_population_num = common.num_population
    index_table = [0] + list(accumulate([int(p * total_population_num) for p in common.selection_rule["proportion"]]))
    cumulative_probability = list(accumulate(common.selection_rule["probability"]))

    offspring_pool = []
    while crossover_num - len(offspring_pool) > 0:
        parent1 = select_parent(total_population, cumulative_probability, index_table)
        parent2 = select_parent(total_population, cumulative_probability, index_table)

        while parent1["model_bit"].values == parent2["model_bit"].values:  # in case self crossover
            parent2 = select_parent(total_population, cumulative_probability, index_table)

        # The number of bit can be differet between parents because each model has different architecture depth.
        p1_depth = parent1["model_length"].values[0]
        p2_depth = parent2["model_length"].values[0]
        p1_actual_bit = list(parent1["model_bit"].values[0][int(p1_depth * common.per_conv_bit_num) * -1:])
        p2_actual_bit = list(parent2["model_bit"].values[0][int(p2_depth * common.per_conv_bit_num) * -1:])
        bit_window_size = min(len(p1_actual_bit), len(p2_actual_bit))

        p1_window, p1_cache = get_bit_window(p1_actual_bit, bit_window_size)
        p2_window, p2_cache = get_bit_window(p2_actual_bit, bit_window_size)

        # perform k-point cross over.
        c1_window, c2_window = p1_window, p2_window
        rnd_k = np.random.randint(low=1, high=4)  # one of [1, 2, 3]
        slice_indices = [0] + sorted(random.sample(range(1, bit_window_size), rnd_k)) + [bit_window_size]
        for idx in range(1, rnd_k + 2):
            start, end = slice_indices[idx - 1], slice_indices[idx]
            c1_window[start:end], c2_window[start:end] = p2_window[start:end], p1_window[start:end]
        p1_actual_bit[p1_cache[0]: p1_cache[1]] = c1_window
        p2_actual_bit[p2_cache[0]: p2_cache[1]] = c2_window
        child1_actual_bit = "".join(p1_actual_bit)
        child2_actual_bit = "".join(p2_actual_bit)
        total_gene = list(total_population["model_bit"].values)
        offspring_pool = add_or_drop_child(child1_actual_bit, offspring_pool, total_gene, common)
        offspring_pool = add_or_drop_child(child2_actual_bit, offspring_pool, total_gene, common)
    return offspring_pool


def mutation(total_population, common):
    mutated_gene = []
    while common.mutation_num - len(mutated_gene) > 0:
        parent = total_population.sample()
        p_depth = parent["model_length"].values[0]
        p_actual_bit = list(parent["model_bit"].values[0][int(p_depth * common.per_conv_bit_num) * -1:])
        # select a random index among actual bits
        rnd_idx = np.random.randint(low=0, high=len(p_actual_bit))
        p_actual_bit[rnd_idx] = '0' if p_actual_bit[rnd_idx] == '1' else '1'
        c_actual_bit = "".join(p_actual_bit)
        total_gene = list(total_population["model_bit"].values)
        mutated_gene = add_or_drop_child(c_actual_bit, mutated_gene, total_gene, common)
    return mutated_gene
