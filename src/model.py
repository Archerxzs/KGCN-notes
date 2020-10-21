import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)#解析参数，并定义聚合器
        self._build_inputs()#定义输入类型，user_indices、item_indices、labels==>（batch_size,1）
        self._build_model(n_user, n_entity, n_relation)#定义计算的模型
        self._build_train()

    @staticmethod
    def get_initializer():
        #该函数返回一个用于初始化权重的初始化程序 “Xavier” 这个初始化器是用来保持每一层的梯度大小都差不多相同。
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        print("_parse_args"+str(adj_entity.shape))

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        #[   0    0    0 ... 1104 1104 1105]  65536个
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        #根据user_indices 可能会重复的话，这个user_embeddings在embedding_lookup的时候，会存在重复的行
        #user_emb_matrix:(138159,32) user_indices:(65536)(eg:[   0    0    0 ... 1104 1104 1105])
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}

        # entities[0]:(65536,1) entities[1]:(65536,4) entities[2]:(65536,16)
        # relations[0]:(65536,4) relations[1]:(65536,16)
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        #[65536,32]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # [batch_size]
        #[65536,32]*[65536,32]==>[65536]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    '''
    参数seeds:为输入的item的下标，可以代表为（花木兰、魁拔、雾山五行、刀剑神域....） 纬度为（65536）
    然后扩展了一次纬度变为（65536，1）
    然后转为entities(list），此时len(entities)=1、len(relations)=0
    再循环（n_iter=2代表了2次迭代，就是找到邻居的邻居（关键！！））    
    '''
    def get_neighbors(self, seeds):#65536
        print("get_neighbors")
        print(seeds.shape)
        seeds = tf.expand_dims(seeds, axis=1)#(65536,1)
        print(seeds.shape)
        entities = [seeds]#len(entities)=1,entities[0]=(65536,1)纬的矩阵
        print(len(entities))
        relations = []
        #n_item=2===>i∈[0,1]
        for i in range(self.n_iter):
            print("第"+str(i)+"次")
            #adj_entity：(102569,4)
            #neighbor_entities：（65536,4）
            #entities[0]:(65536,1)
            #entities[1]:(65536,4)
            #entities[2]:(65536,16)

            #adj_relation：（102569，4）
            #relations[0]:(65536,4)
            #relations[1]:(65536,16)
            '''
            当到i=1的时候，neighbor_entities存的就是entities[0]里的实体的邻居了
            eg:(假设这些邻居都存在）
            entiies[0]=[西游记，三国演义，红楼梦，水浒]
            
            i=0的时候，neighbour_entities=[[孙悟空，吴承恩，小说，神话],[诸葛亮，..],[...],[...]]
            此时entities[1]=[[孙悟空，吴承恩，小说，神话],[...],[...],[...]]
            
            i=1的时候
            neighbour_entities=[[(接下来四个是孙悟空的邻居）西游记，齐天大圣，猴子，佛,（接下来是吴承恩的邻居）作者，文学家，出生日期,...],
                                [(接下来四个是诸葛亮的邻居）卧龙坡，借箭大法，观星,丞相,......],
                                [....],
                                [....]]
            
            i=0:
                tf.gather(self.adj_entity, entities[i])===>（65536，1，4）
                neighbor_entities=tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])===>（65536,4）
                neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])===>（65536,4）
            i=1:
                tf.gather(self.adj_entity, entities[i])===>(65536,4,4)
                neighbor_entities=tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])===>（65536,16）
                neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])===>（65536,16）
            
            '''
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])

            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])

            entities.append(neighbor_entities)

            relations.append(neighbor_relations)

        return entities, relations

    def aggregate(self, entities, relations):
        print("aggregate")

        aggregators = []  # store all aggregators
        #entity_emb_matrix:(102569,32)
        '''
        entity_vectors[0]:(65536, 1, 32)
        entity_vectors[1]:(65536, 4, 32)
        entity_vectors[2]:(65536, 16, 32)
        '''
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]

        '''
        relation_vectors[0]:(65536, 4, 32)
        relation_vectors[1]:(65536, 16, 32)
        '''
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        #i∈[0,1]
        for i in range(self.n_iter):
            if i == self.n_iter - 1:# i==1
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:# i==0
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            #激活函数选择的不一样
            #i=0 relu i=1 tanh
            aggregators.append(aggregator)

            #i=0 entity_vectors_next_iter=[[65536,1,32],[65536,4,32]]
            #i=1 entity_vectors_next_iter=[[65536,1,32]]
            entity_vectors_next_iter = []

            for hop in range(self.n_iter - i):
                '''
                    i=0:hop∈[0,1] 还有邻居的邻居可以算
                    i=1:hop∈[0] 只有邻居可以算
                '''
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                # [65536,1,32]
                # [65536,4,32]
                '''
                    self_vectors:当前的自身向量（西游记）
                    neighbor_vectors：当前的自身向量的邻居的向量（吴承恩）
                    neighbor_relations：当前的自身向量和邻居的关系（作者）
                '''
                vector = aggregator(
                                    #entity_vectors[0]:(65536, 1, 32)
                                    #entity_vectors[1]:(65536, 4, 32)
                                    self_vectors=entity_vectors[hop],
                                    #entity_vectors[1]:(65536, 4, 32)===>(65536, 1,4, 32)
                                    #entity_vectors[2]:(65536, 16, 32)===>(65536, 4,4, 32)
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    #relation_vectors[0]:(65536, 4, 32)==>(65536,1,4,32)
                                    #relation_vectors[1]: (65536, 16, 32)==>(65536,4,4,32)
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    #[65536,32]
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            # i=0 entity_vectors=[[65536,1,32],[65536,4,32]]
            # i=1 entity_vectors=[[65536,1,32]]
            #entity_vectors在改变
            #经过多次聚合得到entity的最终向量
            entity_vectors = entity_vectors_next_iter

        #[65536,32]
        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)

        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
