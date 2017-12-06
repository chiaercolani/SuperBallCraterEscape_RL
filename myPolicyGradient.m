classdef myPolicyGradient <handle
    %MYPOLICYGRADIENT This class contains all the functions for the
    %Reinforcement Learning process in MATLAB
    %   Detailed explanation goes here
    
    properties
      learningRate      % Learning rate of the model
      layersDims        % Dimensions of the layers
      gamma             % Decay rate for the rewards
      L                 % Number of layers in the network
      
      epObservations    % Observations in one episode
      epActions         % Actions in each episode
      epRewards         % Rewards in each episode
      ep
      
      % Weights and biases
      W1
      b1
      W2
      b2
      
      %hidden layer forward prop
      hidden
      
    end
    
    methods
        
        %Constructor for the Agent
        function obj=myPolicyGradient(hiddenSize,observations,actionSize,gamma)
            
            obj.layersDims=[size(observations,1) hiddenSize actionSize];
            obj.gamma=gamma;
            obj=initializeParameters(obj);
            
            %Initialize lists for observations,actions and rewards
            obj.epObservations=[];
            obj.epActions=[];
            obj.epRewards=[];
            
        end
        
        %Choose an action to perform based on a probability distribution
        %given by the forward propagation
        function actions=pickActions(obj,observations)
            
            actionsProb=forwardPropagation(obj,observations);
            
            %randomly pick action
            possibleActions=[0 1 2];
            
            %Action 0 is not to move
            actions=zeros(24,1);
            
            for i=1:24
                actions(i)=randsample(possibleActions,1,true,actionsProb(i,:));
            end
            
            
        end
        
        %Store the transition
        function storeTransition(obj,state,action,reward)
            obj.epObservations=[obj.epObservations state];    
            obj.epActions=[obj.epActions action];       
            obj.epRewards=[obj.epRewards reward];
        end
        
        
        %Initialize W and b
        function obj=initializeParameters(obj)
            
            % fro now layersDims contains the dimensions of the input,
            % hidden and output layer. TODO: make multilayering possible
            %obj.L= length(obj.layersDims);
            
            %Create weigths and biases for hidden layer
            obj.W1=normrnd(0,1,[obj.layersDims(1),obj.layersDims(2)]);
            obj.b1=zeros(1,obj.layersDims(2));
            
            %Create W and B for output layer
            obj.W2=normrnd(0,1,[obj.layersDims(2),obj.layersDims(3)]);
            obj.b2=zeros(1,obj.layersDims(3));
            
        end
        
        %Forward propagation. TODO: extend to multilayering
        function actionsProb=forwardPropagation(obj,observations)
            
            %LINEAR->RELU->LINEAR->SOFTMAX
            % compute first hidden layer (RELU=poslin)
            hidden=poslin(observations'*obj.W1+obj.b1);

            % compute output layer
            actions=hidden*obj.W2+obj.b2;
            % sigmoid to find the action probabilities
            actionsProb=softmax(actions',1);
            actionsProb=actionsProb';
            
        end
        
        function costs=costComputation()
            
            
            
        end
        
        function gradients=backwardPropagation()
            
            
        end
        
        function parameters=updateParameters()
            
        end
        
        
        
        function nnLearn(obj)
            discountedRewards=computeDiscountedRewards(obj);
            
            %Backpropagation + update parameters
            
            %clear episode values
            
            obj.epObservations=[];
            obj.epActions=[];
            obj.epRewards=[];
                      
        end
        
        function discountedRewards=computeDiscountedRewards(obj)
            
            discR=zeros(shape(obj.epRewards));
            runningAdd=0;
            for t=shape(obj.epRewards,1):-1:1
                runningAdd=runningAdd*obj.gamma+obj.epRewards(t);
                discR(t)=runningAdd;
            end
            %Normalize discounted rewards
            discR=discR-mean(discR);
            discountedRewards=discR/std(discR);
                      
        end
        
    end
    
end

