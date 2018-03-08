
import numpy as np
import tensorflow as tf


class agent_class:

    def __init__(self,learning_rate,actions_size,hidden_layer_size,features_size,gamma,L):

        self.lr=learning_rate
        self.a_size=actions_size
        self.hl_size=hidden_layer_size
        self.f_size=features_size
        self.gamma=gamma
        self.L=L
        self.motorNumber=24

        self.ep_observations, self.ep_actions, self.ep_rewards = [], [], []

        #INITALIZE NEURAL NETWORK

        #Initialize placeholders and variables for NN
        self.reward_holder = tf.placeholder(shape=[None,],dtype=tf.float32,name="reward_holder")
        self.action_holder = tf.placeholder(shape=[None,],dtype=tf.int32,name="action_holder")
        self.nn_features=tf.placeholder(tf.float32,[None,self.f_size],name="nn_features")

        # Initialize weights and biases
        with tf.name_scope('init'):
            Wx = tf.get_variable("Wx", shape=[self.f_size, self.hl_size],initializer=tf.contrib.layers.xavier_initializer())
            Wy = tf.get_variable("Wy", shape=[self.hl_size, self.a_size],initializer=tf.contrib.layers.xavier_initializer())
            bh = tf.Variable(tf.zeros([1,self.hl_size]));
            by = tf.Variable(tf.zeros([1,self.a_size]));

        # Hidden Layer of the NN: RELU
        #TODO extend this to Deep NN (Will also have to generate more W and b)
        hidden1=tf.nn.relu(tf.matmul(self.nn_features,Wx)+bh)

        # Output Layer of NN: Linear
        actions=tf.matmul(hidden1,Wy)+by
        
        # Compute probabilities with softmax function
        self.actions_prob=tf.nn.softmax(actions,name='actions_prob')

        #log_prob=[]
        #for i in range(self.motorNumber):
        #    log_prob = np.append(log_prob,tf.nn.sparse_softmax_cross_entropy_with_logits(logits=actions, labels=self.action_holder[:,i]))


        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=actions, labels=self.action_holder)

        # Loss function
        loss=tf.reduce_mean(log_prob*self.reward_holder)

        # Optimizer
        self.optimizer=tf.train.AdamOptimizer(self.lr).minimize(loss)

        # Create TF session
        self.sess=tf.Session()

        # Run the session and initialize all the variables
        self.sess.run(tf.global_variables_initializer())

    def sessionClose(self):
        self.sess.close()

    def pick_action(self,feature):
        # Run the NN and generate the probability of picking each action
        prob_dist=self.sess.run(self.actions_prob,feed_dict={self.nn_features:feature})

        # Pick which action to perform with probability=prob_dist
        action=[]
        for i in range(prob_dist.shape[0]):
            action =np.append(action, np.random.choice(3,1, p=prob_dist[i,:]))
        return action

    def cancel_transition(self):
        # Initialize episode observations, actions and rewards
        self.ep_observations, self.ep_actions, self.ep_rewards = [], [], []

    def store_transition(self,state,action,reward):
        # Save observations, actions and rewards for each transition
        self.ep_observations.append(state)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)

    def computeDiscountedRewards(self):

        # Compute discounted rewards using gamma
        discounted_r = np.zeros_like(self.ep_rewards)
        #avoid having the discounted rewards value being 0
        running_add = 0.00000001
        for t in reversed(range(0, len(self.ep_rewards))):
            running_add = running_add * self.gamma + self.ep_rewards[t]
            discounted_r[t] = running_add

        #Normalize the rewards
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)

        return discounted_r

    def nn_learn(self,rewards):

        # Stack horizontally the inputs, reshape the outputs to match the inputs and duplicate the rewards so that all outputs of the same
        # episode receive the same reward
        self.sess.run(self.optimizer,feed_dict={self.nn_features: np.vstack(self.ep_observations),self.action_holder: np.reshape(self.ep_actions,np.array(self.ep_actions).shape[0]*np.array(self.ep_actions).shape[1]),self.reward_holder: np.repeat(rewards,24)})

        # Initialize episode observations, actions and rewards after learning
        self.ep_observations, self.ep_actions, self.ep_rewards = [], [], []
