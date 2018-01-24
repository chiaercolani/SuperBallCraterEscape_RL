
import numpy as np
import tensorflow as tf
import os


class agent_class:

    def __init__(self,learning_rate,actions_size,hidden_layer_size,features_size,gamma,L,n,m):

        self.lr=learning_rate
        self.a_size=actions_size
        self.hl_size=hidden_layer_size
        self.f_size=features_size
        self.gamma=gamma
        self.L=L
        self.motorNumber=24
        self.n=n
        self.dir= os.path.dirname(os.path.realpath(__file__))
        self.m=m
        self.ep_observations, self.ep_actions, self.ep_rewards = [], [], []

        #INITALIZE NEURAL NETWORK
        #graphName= "g"+str(self.n)
        #graphName=tf.Graph()
        self.graph=tf.Graph()
        #self.sess=tf.Session(self.graph)

        with self.graph.as_default():
            #Initialize placeholders and variables for NN
            self.reward_holder = tf.placeholder(shape=[None,],dtype=tf.float32,name="reward_holder")
            self.action_holder = tf.placeholder(shape=[None,],dtype=tf.int32,name="action_holder")
            self.nn_features=tf.placeholder(tf.float32,[None,self.f_size],name="nn_features")

            # Initialize weights and biases
            with tf.name_scope('init'):
                Wx = tf.get_variable("Wx_"+str(self.n)+"_"+str(self.m), shape=[self.f_size, self.hl_size],initializer=tf.contrib.layers.xavier_initializer())
                Wy = tf.get_variable("Wy_"+str(self.n)+"_"+str(self.m), shape=[self.hl_size, self.a_size],initializer=tf.contrib.layers.xavier_initializer())
                bh = tf.Variable(tf.zeros([1,self.hl_size]),name="bh_"+str(self.n)+"_"+str(self.m));
                by = tf.Variable(tf.zeros([1,self.a_size]),name="by_"+str(self.n)+"_"+str(self.m));


            # Hidden Layer of the NN: RELU
            hidden1=tf.nn.relu(tf.matmul(self.nn_features,Wx)+bh)

            # Output Layer of NN: Linear
            actions=tf.matmul(hidden1,Wy)+by

            # Compute probabilities with softmax function
            self.actions_prob=tf.nn.softmax(actions,name='actions_prob')

            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=actions, labels=self.action_holder)

            # Loss function
            loss=tf.reduce_mean(log_prob*self.reward_holder)

            # Optimizer
            self.optimizer=tf.train.AdamOptimizer(self.lr).minimize(loss)

            #Define saver object to save NN
            self.saver=tf.train.Saver()

            # Create TF session
            self.sess=tf.Session(graph=self.graph)

            # Run the session and initialize all the variables
            self.sess.run(tf.global_variables_initializer())

    def initializeVars(self):
        self.sess.run(tf.global_variables_initializer())

    def sessionClose(self):
        self.sess.close()

    def pick_action(self,feature):
        # Run the NN and generate the probability of picking each action
        prob_dist=self.sess.run(self.actions_prob,feed_dict={self.nn_features:feature})
        #print(prob_dist)
        # Pick which action to perform with probability=prob_dist
        action=[]
        for i in range(prob_dist.shape[0]):
            action =np.append(action, np.random.choice(3,1, p=prob_dist[i,:]))
        return action

    def pickActionHighestProb(self,feature):
        # Run the NN and generate the probability of picking each action
        prob_dist=self.sess.run(self.actions_prob,feed_dict={self.nn_features:feature})
        prob_dist=np.reshape(prob_dist,(3))

        # Pick which action to perform with probability=prob_dist

        action =np.argmax(prob_dist)

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
        #self.sess.run(self.optimizer,feed_dict={self.nn_features: np.vstack(self.ep_observations),self.action_holder: np.reshape(self.ep_actions,np.array(self.ep_actions).shape[0]*np.array(self.ep_actions).shape[1]),self.reward_holder: np.repeat(rewards,24)})
        self.sess.run(self.optimizer,feed_dict={self.nn_features: np.vstack(self.ep_observations),self.action_holder: np.array(self.ep_actions),self.reward_holder: rewards})
        # Initialize episode observations, actions and rewards after learning
        self.ep_observations, self.ep_actions, self.ep_rewards = [], [], []


    def saveSession(self):
        self.saver.save(self.sess,self.dir+'/MultiBruteForceResults/network_'+str(self.n)+'_'+str(self.m))

    def loadSession(self):
        self.saver = tf.train.import_meta_graph(self.dir+'/MultiBruteForceResults/network_'+str(self.n)+'_'+str(self.m)+'.meta')
        self.saver.restore(self.sess,self.dir+'/MultiBruteForceResults/network_'+str(self.n)+'_'+str(self.m))
        # Get saved graph
        #self.graph=tf.get_default_graph()
        #print(self.graph.get_operations())
        #Code to print the variables
        #with self.graph.as_default():
        #    variables_names =[v.name for v in tf.trainable_variables()]
        #    values = self.sess.run(variables_names)
        #    for k,v in zip(variables_names, values):
        #        print(k, v)

    def runSession(self,observations):
        self.sess.run(feed_dict={self.nn_features:observations})

    def saveCoordinates(self,coordinates):
        #np.savetxt(self.dir+'/results/rewards.txt','NEW LINE')
        np.savetxt(self.dir+'/MultiBruteForceResults/coordinates.txt',coordinates)
        #r=rewards.tolist())
