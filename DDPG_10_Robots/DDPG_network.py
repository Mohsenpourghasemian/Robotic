import os
import tensorflow as tf
import numpy as np
from tensorflow.initializers import random_uniform
from tensorflow.keras import layers


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.5, theta=0.5, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta *(self.mu-self.x_prev)*self.dt + \
            self.sigma*np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ =self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.model = 0
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name + 'ddpg_ckpt')
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradients)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):

        with tf.variable_scope(self.name):
            # Initialize weights between -3e-5 and 3-e5
            last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)
            self.action_gradients = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='gradients')

            # Actor will get observation of the agent
            # not the observation of other agents
            inputs = layers.Input(shape=[None, *self.input_dims])
            out = layers.Dense(32, activation="tanh", kernel_initializer="lecun_normal")(inputs)
            out1 = layers.BatchNormalization()(out)
            out2 = layers.Dense(32, activation="tanh", kernel_initializer="lecun_normal")(out1)
            out3 = layers.BatchNormalization()(out2)
            out4 = layers.Dense(32, activation="tanh", kernel_initializer="lecun_normal")(out3)
            out5 = layers.BatchNormalization()(out4)

            # Using tanh activation as action values for
            # for our environment lies between -1 to +1
            outputs = layers.Dense(self.n_actions, activation="tanh", kernel_initializer=last_init)(out5)

            self.mu = outputs
            self.model = tf.keras.Model(inputs, outputs)
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
#
            # self.action_gradients = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='gradients')
            # f1 = 1. / np.sqrt(self.fc1_dims)
            # dense1 = tf.layers.dense(self.input, units=self.fc1_dims, kernel_initializer=random_uniform(-f1, f1),
            #                          bias_initializer=random_uniform(-f1, f1))
            # batch1 = tf.layers.batch_normalization(dense1)
            # layer1_activation = tf.nn.tanh(batch1)
            # f2 = 1. / np.sqrt(self.fc2_dims)
            # dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims, kernel_initializer=random_uniform(-f2, f2),
            #                          bias_initializer=random_uniform(-f2, f2))
            # batch2 = tf.layers.batch_normalization(dense2)
            # layer2_activation = tf.nn.tanh(batch2)
#
            # f3 = 1. / np.sqrt(self.fc2_dims)
            # dense3 = tf.layers.dense(layer2_activation, units=32, kernel_initializer=random_uniform(-f3, f3),
            #                          bias_initializer=random_uniform(-f3, f3))
            # batch3 = tf.layers.batch_normalization(dense3)
            # layer3_activation = tf.nn.tanh(batch3)
            # # f3 = 0.003
            # # mu1 = tf.layers.dense(layer3_activation, units=140, activation='sigmoid',
            # #                     kernel_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3))
            # mu1 = tf.layers.dense(layer3_activation, units=self.n_actions, activation='tanh', kernel_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3))
            # # mu1 = tf.multiply(mu1, 1)
            # #mu2 = tf.layers.dense(layer2_activation, units=10, activation='softmax',
            # #                   kernel_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3))
            # #self.mu = tf.concat([mu1, mu2], 1, name='concat')
            # # self.mu = tf.multiply(mu, self.action_bound)
            # # self.model = tf.keras.Model(self.input, mu1)
            # self.mu=mu1

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input:inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action_gradients: gradients})

    def load_checkpoint(self):
        print("...loading ...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...saving...")
        self.saver.save(self.sess, self.checkpoint_file)


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims,
                 batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name + 'ddpg_ckpt')
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                                  shape=[None, self.n_actions],
                                                  name='actions')
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, 1],
                                           name='targets')
            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)
            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            action_in = tf.layers.dense(self.actions, units=self.fc2_dims,
                                        activation='relu')
            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)
            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1,
                                     kernel_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.actions: actions})

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.actions: actions,
                                                       self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients, feed_dict={self.input: inputs, self.actions: actions})

    def load_checkpoint(self):
        print("...loading ...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...saving...")
        self.saver.save(self.sess, self.checkpoint_file)


class Agent(object):
    def __init__(self, name_actor, name_critic, name_target_actor, name_target_critic, alpha, beta, input_dims, tau, gamma=0.0, n_actions=7, max_size=100000, layer1_size=32, layer2_size=32, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()

        self.actor = Actor(alpha, n_actions, name_actor, input_dims, self.sess,
                           layer1_size, layer2_size, 1, batch_size=64, chkpt_dir=chkpt_dir)
        self.critic = Critic(beta, n_actions, name_critic, input_dims, self.sess,
                             layer1_size, layer2_size, chkpt_dir=chkpt_dir)

        self.target_actor = Actor(alpha, n_actions, name_target_actor, input_dims, self.sess,
                           layer1_size, layer2_size, 1, batch_size=64, chkpt_dir=chkpt_dir)
        self.target_critic = Critic(beta, n_actions, name_target_critic, input_dims, self.sess,
                             layer1_size, layer2_size, chkpt_dir=chkpt_dir)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.update_critic = \
        [self.target_critic.params[i].assign(
                        tf.multiply(self.critic.params[i], self.tau) \
                    + tf.multiply(self.target_critic.params[i], 1.0 - self.tau))
        for i in range(len(self.target_critic.params))]
        self.update_actor = \
        [self.target_actor.params[i].assign(
                        tf.multiply(self.actor.params[i], self.tau) \
                    + tf.multiply(self.target_actor.params[i], 1.0 - self.tau))
        for i in range(len(self.target_actor.params))]
        self.sess.run(tf.global_variables_initializer())
        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=True):
        if first:
            old_tau = self.tau
            self.tau = 0.001
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def noise(self):
        return self.noise()

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        mu_prime = mu #+ self.noise()
        # mu_prime = np.clip((mu + 1) / 2, 0., 1.)
        # print("\n\n")
        # print("mu    ", mu_prime, "  \n\n mu[0]", mu_prime[0])
        # print("--------------------------------------------------------------")
        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = np.reshape(target, (self.batch_size, 1))
        _ = self.critic.train(state, action, target)
        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)

        self.actor.train(state, grads[0])
        self.update_network_parameters(first=True)

    def save_model(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def get_param(self):
        print("len(self.actor.params)     ", len(self.actor.params))
        return self.actor.params

    def set_params(self, actor_params):
        # self.sess.run(tf.global_variables_initializer())
        self.actor.sess.run([self.actor.params[i].assign(tf.multiply(actor_params[i], 1)) for i in range(len(self.actor.params))])


class Federated_Server(object):
    def __init__(self, name_actor, name_critic, input_dims, n_actions=7, layer1_size=32, layer2_size=32):
        self.sess = tf.Session()
        self.actor = Actor(1, n_actions, name_actor, input_dims, self.sess,
                           layer1_size, layer2_size, 1)
        # self.sess.run(tf.global_variables_initializer())

    def federation(self, actor_params, actor_params1):

        self.actor.sess.run([self.actor.params[i].assign((tf.multiply(actor_params[i], 1)+
                                                         tf.multiply(actor_params1[i], 1))/2) for i in range(len(self.actor.params))])
        return self.actor.params


class Federated_Server_AP(object):
    def __init__(self, name_actor, name_critic, input_dims, n_actions=8, layer1_size=32, layer2_size=32):
        self.sess = tf.Session()
        self.actor = Actor(1, n_actions, name_actor, input_dims, self.sess,
                           layer1_size, layer2_size, 1)
        # self.sess.run(tf.global_variables_initializer())

    def federation(self, actor_params1, actor_params2, actor_params3, actor_params4,):
        self.actor.sess.run([self.actor.params[i].assign((tf.multiply(actor_params1[i], 1)+
                                                         tf.multiply(actor_params2[i], 1)+
                                                         tf.multiply(actor_params3[i], 1)+
                                                         tf.multiply(actor_params4[i], 1))/4) for i in range(len(self.actor.params))])
        return self.actor.params