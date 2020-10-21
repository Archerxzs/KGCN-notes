import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        # neighbor_vectors[1]:(65536, 4, 32)===>(65536, 1,4, 32)
        # neighbor_vectors[2]:(65536, 16, 32)===>(65536, 4,4, 32)

        # relation_vectors[0]:(65536, 4, 32)==>(65536,1,4,32)
        # relation_vectors[1]: (65536, 16, 32)==>(65536,4,4,32)
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            #user_embeddings:[65536,32]==>[65536,1,1,32]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            #[65536,1,1,32]*[65536,1,4,32]===>[65536,1,4]
            #[65536,1,1,32]*[65536,4,4,32]===>[65536,4,4]
            #关系r对用户u的重要性得分
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)

            #[65536,1,4]
            #[65536,4,4]
            #把得分归一化
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1]===>[65536,1,4,1]
            # [batch_size, -1, n_neighbor, 1]===>[65536,4,4,1]
            #扩列
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            #[65536,1,4,1]*[65536,1,4,32]===>[65536,1,32]
            #[65536,4,4,1]*[65536,4,4,32]===>[65536,4,32]
            #每个邻居的信息和该邻居的关系对用户的关系得分内积，得到邻域信息的聚合
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        #variable_scope：管理同名的参数
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        # [65536,1,32]
        # [65536,4,32]
        #得到了所有邻居对自己的补充信息
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        #(65536, 1, 32)+[65536,1,32]===>[65536,1,32]===>[65536,32]
        #(65536, 4, 32)+[65536,4,32]===>[65536,4,32]===>[262144,32]
        #把邻居信息和自身信息用sum聚合器聚合
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        #[65536,1,32]
        #[65536,4,32]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)
