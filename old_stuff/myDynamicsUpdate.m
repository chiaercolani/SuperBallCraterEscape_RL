function myDynamicsUpdate(tensStruct1, dynamicsPlot1, displayTimeInterval,agent1,observations1)
% This function will perform dynamics update each timestep.

%create some persistent variables for objects and structs
persistent tensStruct dynamicsPlot tspan i agent observations

if nargin>1
    i = 0;
    tensStruct = tensStruct1;
    dynamicsPlot = dynamicsPlot1;
    tspan = displayTimeInterval;
    agent=agent1;
    observations=observations1;
end

deltaSpool=0.05;

%Define the number of maximum cycles before resetting the episode
max_ep_cycle=30;

%Initialize episode rewards and done signal
reward=0;

%Define done signal-> done when no rod is touching the ground
% TODO improve the reward systemm (bigger reward if the robot lifts for
% longer periods of time
done=0;
i=0;

%Wait for superBall to stabilize
    while i<20

         % Update nodes:
        dynamicsUpdate(tensStruct, tspan);
        dynamicsPlot.nodePoints = tensStruct.ySim(1:end/2,:);
        updatePlot(dynamicsPlot);

        drawnow  %plot it up
        i=i+1;
    end

   for k= 1:max_ep_cycle

        actions=pickActions(agent,observations');
        spoolingDistance=zeros(24,1);
        motorsToMove=actions>0;
        for i=1:24
            %actions=1 means that rest length of string has to increase
            if actions(i)==1
                spoolingDistance(i)=deltaSpool;
            %actions=2 means that rest length of string has to increase
            elseif actions(i)==2
                spoolingDistance(i)=-deltaSpool;
            end
        end
        %Apply actions
        tensStruct.simStruct.stringRestLengths(motorsToMove) = tensStruct.simStruct.stringRestLengths(motorsToMove)+spoolingDistance(motorsToMove);

        % Update nodes:
        dynamicsUpdate(tensStruct, tspan);
        dynamicsPlot.nodePoints = tensStruct.ySim(1:end/2,:);
        updatePlot(dynamicsPlot);

        drawnow  %plot it up


        %Get new rest lengths after action is performed
        observations=tensStruct.simStruct.stringRestLengths;

        %Check done signal and reward TBD better
        if dynamicsPlot.plotErrorFlag==1
            disp('Error in config');
            break;            
        elseif tensStruct.rewardTouchingGnd==logical(zeros(12,1))
            reward=10;
            done=1;
        end

        %Store the transition
        storeTransition(agent,observations,actions,reward);

        if done==1
            %Sum rewards of the episode
            epRewardSum=sum(agent.epRewards);
            fprintf('Lift SuperBall %d cycles\n',k);
            break;
        end
        
   end

end


