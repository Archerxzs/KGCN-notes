# import os
# import numpy as np
#
# def dataset_split(rating_np):
#     print('splitting dataset ...')
#     ratio=1
#     # train:eval:test = 6:2:2
#     eval_ratio = 0.2
#     test_ratio = 0.2
#     n_ratings = rating_np.shape[0]
#
#     #随机生产评估集
#     eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
#     left = set(range(n_ratings)) - set(eval_indices)
#     test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
#     train_indices = list(left - set(test_indices))
#     if ratio < 1:
#         train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * ratio), replace=False)
#
#     train_data = rating_np[train_indices]
#     eval_data = rating_np[eval_indices]
#     test_data = rating_np[test_indices]
#
#     return train_data, eval_data, test_data
#
# def load_rating():
#     print('reading rating file ...')
#
#     # reading rating file
#     rating_file = '../data/movie/ratings_final'
#     if os.path.exists(rating_file + '.npy'):
#         rating_np = np.load(rating_file + '.npy')
#     else:
#         rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
#         np.save(rating_file + '.npy', rating_np)
#
#     n_user = len(set(rating_np[:, 0]))
#     n_item = len(set(rating_np[:, 1]))
#     train_data, eval_data, test_data = dataset_split(rating_np)
#
#     return n_user, n_item, train_data, eval_data, test_data
#
# def get_feed_dict(model, data, start, end):
#     feed_dict = {model.user_indices: data[start:end, 0],
#                  model.item_indices: data[start:end, 1],
#                  model.labels: data[start:end, 2]}
#     return feed_dict
#
# n_user, n_item, train_data, eval_data, test_data = load_rating()
# '''
# <class 'int'>
# 138159
# <class 'numpy.ndarray'>
# 8100974
# (8100974, 3)
# <class 'numpy.ndarray'>
# 65535
# (65535,)
#
# [  0 770   1]
# 770
# 1795
# [ 770 1795 3842 ...  663 4480  773]
# (65535, 1)
# [<tf.Tensor 'ExpandDims:0' shape=(65535, 1) dtype=int64>]
# 1
# <class 'list'>
# '''
# print("1")
# print(str(type(n_user)))
# print(str(n_user))
# print(str(type(train_data)))
# print(str(len(train_data)))
# print(str(train_data.shape))
# print(str(type(train_data[0:65535, 1])))
# print(str(len(train_data[0:65535, 1])))
# print(str(train_data[0:65535, 1].shape))
# print()
#
# print("2")
# print(train_data[0])
# print(train_data[0:65535, 1][0])
# print(train_data[0:65535, 1][1])
# print(train_data[0:65535, 1])
# import tensorflow as tf
# seeds=tf.expand_dims(train_data[0:65535, 1], axis=1)
# print(str(seeds.shape))
# print()
#
# print("3")
# entities = [seeds]
# print(entities)
# print(entities[0])
# print(str(len(entities)))
# print(str(type(entities)))
# # print(str(len(entities[0])))
# print(str(type(entities[0])))
# print(str(entities[0].shape))
#
#
# adj_entity=np.random.randint(1,10,[102569,4])
# temp1=tf.gather(adj_entity, entities[0])
# print()
# print("4")
# # print(str(len(temp1)))
# print(str(type(temp1)))
# print(str(temp1.shape))
# temp2=tf.reshape(temp1, [65535, -1])
# print(str(type(temp2)))
# print(str(temp2.shape))
# entities.append(temp2)
#
# print()
# print("5")
# temp1=tf.gather(adj_entity, entities[1])
# print(str(type(temp1)))
# print(str(temp1.shape))
# temp2=tf.reshape(temp1, [65535, -1])
# print(str(type(temp2)))
# print(str(temp2.shape))
# entities.append(temp2)
#
# print(str(len(entities)))
# print(str(entities[0].shape))
# print(str(entities[1].shape))
# print(str(entities[2].shape))

# import tensorflow as tf
# import numpy as np
# p = tf.Variable(tf.random_normal([102569,32]))
# q = np.random.randint(1,10,[65536,16])
# b = tf.nn.embedding_lookup(p, q)  # 查找张量中的序号为1和3的
# print(b.shape)
# print(b)
# print()
#
# entities=[]
# entities.append(np.random.randint(1,10,[65536,1]))
# entities.append(np.random.randint(1,10,[65536,4]))
# entities.append(np.random.randint(1,10,[65536,16]))
# entity_vectors = [tf.nn.embedding_lookup(p, i) for i in entities]
# print(len(entity_vectors))
# print(entity_vectors[0].shape)
# print(entity_vectors[1].shape)
# print(entity_vectors[2].shape)

# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     print(sess.run(b))
# #     # print(c)
# #     print(sess.run(p))
# #     print(p)
# #     print(type(p))

# import tensorflow as tf
# import numpy as np
# sess = tf.Session()
# t=np.random.randint(1,10,[4,5])
# a=np.random.randint(1,10,[2,1])
# g2=tf.gather(t,a)
# print(t)
# print(a)
# print(sess.run(g2))

# shape = [65536, -1, 4, 32]
# print(type(shape))
# print(len(shape))
# print(shape[0])

# import tensorflow as tf
# a = tf.constant([[1],[3,4]])
# b = tf.constant([[1,1],[2,2]])
# sess = tf.Session()
# print(sess.run(a*b))
import numpy as np
import tensorflow as tf
adj_entity = np.arange(400).reshape(100,4)
adj_entity[0]=np.array([13,19,18,21])
# np.random.shuffle(adj_entity)
print(adj_entity)
print()
entities=[[0,1,2]]
print(entities)
print()
session=tf.Session()
neighbor_entities = tf.reshape(tf.gather(adj_entity, entities[0]), [3, -1])
print(session.run(neighbor_entities))
entities.append(neighbor_entities)
print(entities)
print()
neighbor_entities = tf.reshape(tf.gather(adj_entity, entities[1]), [3, -1])
print(session.run(neighbor_entities))
entities.append(neighbor_entities)
print()
entity_emb_matrix = tf.get_variable(
            shape=[100, 32], initializer=tf.contrib.layers.xavier_initializer(), name='entity_emb_matrix')
entity_vectors = [tf.nn.embedding_lookup(entity_emb_matrix, i) for i in entities]
print(entity_vectors)
