from functions.utils import CommonProperty, train_and_test, fitness_fn, crossover, mutation, adopt_datapipeline_base
from functions.data_pipeline import DataPipeline
from functions.model import Ancestor, Offspring
from silence_tensorflow import silence_tensorflow
import pandas as pd
import time

silence_tensorflow()
import tensorflow as tf
import numpy as np

continue_from_last_generation = True

# SGA settings
model_depth_range = [1, 9]
conv_size_range = [1, 5]
conv_depth_range = [1, 64]
conv_stride_range = [1, 2]
selection_rule = {"proportion": [0.1, 0.3, 0.6],  # the sum should be 1.0
                  "probability": [0.7, 0.2, 0.1]}
mutation_proportion = 0.01
best_parent_proportion = 0.1
num_population = 100
total_generation = 50

# CNN settings
batch = 200
num_classes = 10
max_epoch = 10
max_lr = 0.01
min_lr = 0.00001

# input data settings
train_data_dir = "dataset/mnist/train"
test_data_dir = "dataset/mnist/test"

# Build base datapipeline in graph for resue in different models
data_graph = tf.Graph()
with data_graph.as_default():
    train_datapipe_base = DataPipeline(train_data_dir, batch, is_train=True)
    test_datapipe_base = DataPipeline(test_data_dir, batch, is_train=False)

best_parent_num = int(num_population * best_parent_proportion)  # pass top n% parents to next generations
mutation_num = int(num_population * mutation_proportion)
common = CommonProperty(model_depth_range,
                        conv_size_range,
                        conv_depth_range,
                        conv_stride_range,
                        selection_rule,
                        best_parent_num,
                        mutation_num,
                        num_classes)

total_population = pd.DataFrame(columns=["name", "generation", "accuracy", "parameter", "fitness", "model_length", "model_bit"])
# loop for filling initial population
tic = time.time()
for i in range(num_population):
    # Reuse the base datapipeline in each for loop
    # We may need to build same data pipeline at every iteration if we do not adopt the base datapipeline
    tf.reset_default_graph()
    train_data, test_data = adopt_datapipeline_base(data_graph)
    with tf.Session() as sess:
        name = "model_%03d" % i
        model = Ancestor(train_data, common, name)
        current_epoch = tf.placeholder(tf.float32)
        lr = max_lr - (max_lr - min_lr) / max_epoch * current_epoch
        accuracy = train_and_test(sess, max_epoch, train_data, test_data, model, lr, current_epoch)
        new_entry = {"name": model.name,
                     "generation": 0,
                     "accuracy": accuracy,
                     "parameter": model.total_parameters,
                     "model_length": model.model_length,
                     "model_bit": model.model_bit}
        total_population = total_population.append(new_entry, ignore_index=True)

# calculate fitness and sort
total_population = fitness_fn(total_population)
total_population.to_csv("./generation_log/Ancestor.csv")

best_gene = pd.DataFrame(columns=["name", "generation", "accuracy", "parameter", "fitness", "model_length", "model_bit"])
for generation in range(1, total_generation + 1):
    best_parents = total_population.drop(total_population.index[best_parent_num::])  # keep all the best parents
    best_gene = best_gene.append(best_parents)
    best_gene.to_csv("best_gene.csv")
    offspring_gene_pool = crossover(total_population, common) + mutation(total_population, common)
    best_parent_names = list(total_population["name"].values)[:common.best_parent_num]
    unique_offspring_names = [name for name in ["model_%03d" % i for i in range(num_population + 1)] if name not in best_parent_names]
    offspring_population = pd.DataFrame(columns=["name", "accuracy", "parameter", "fitness", "model_length", "model_bit"])
    for offspring_gene in offspring_gene_pool:
        tf.reset_default_graph()
        train_data, test_data = adopt_datapipeline_base(data_graph)
        with tf.Session() as sess:
            name = unique_offspring_names.pop(0)
            model = Offspring(train_data, offspring_gene, common, name)
            current_epoch = tf.placeholder(tf.float32)
            lr = max_lr - (max_lr - min_lr) / max_epoch * current_epoch
            accuracy = train_and_test(sess, max_epoch, train_data, test_data, model, lr, current_epoch)
            new_entry = {"name": model.name,
                         "generation": generation,
                         "accuracy": accuracy,
                         "parameter": model.total_parameters,
                         "model_length": model.model_length,
                         "model_bit": model.model_bit}
            offspring_population = offspring_population.append(new_entry, ignore_index=True)
    # calculate fitness and sort
    total_population = fitness_fn(pd.concat([best_parents, offspring_population]))
    total_population.to_csv("./generation_log/Generation%03d.csv" % generation)
