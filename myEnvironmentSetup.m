classdef myEnvironmentSetup <handle
    %MYENVIRONMENTSETUP This class contains functions used to emulate
    %SuperBall's escape from a crater. They are used by the Python's
    %reinforcement learning algorithm to get information regarding the
    %environment.
    %   Detailed explanation goes here
    
    properties
        superBall
        superBallDynamicsPlot
        actionSize
        observationSize
        rewards
        
        nodes
        strings
        stringstiffness
        bars
        K
        barStCoeff
        delT
        
        %Initial parameters that can be initialized from Python
        tspan
        wallPos
        wallNeg
        wallHeight
        cDamp
        deltaSpool
    end
    
    methods
        function env=myEnvironmentSetup(tspan,wallPosition,wallHeight,deltaSpool,delT)
            addpath('tensegrityObjects')

            % Inputs from user
            env.tspan=tspan;
            env.deltaSpool=deltaSpool;
            
            %Costumizable parameters
            env.cDamp=100; % String damping coeff
            env.K = 1000;  % String stiffness coeff
            env.barStCoeff=100000; % Bar stiffness coeff
            env.delT=delT; % Delta Time
            
            
            % Wall 1 and 4 are positive values
            env.wallPos=wallPosition;
            env.wallNeg=-wallPosition;
            env.wallHeight=wallHeight;
            barLength = 1.7; %might break things, original 1.7
            barSpacing = barLength / 4; %0.375;
            env.bars = [1:2:11; 
                    2:2:12];
            env.strings = [1  1   1  1  2  2  2  2  3  3  3  3  4  4  4  4  5  5  6  6  7  7  8  8;
                       7  8  10 12  5  6 10 12  7  8  9 11  5  6  9 11 11 12  9 10 11 12  9 10];

            
            env.stringstiffness = env.K*ones(24,1); % String stiffness (N/m)
            barStiffness = env.barStCoeff*ones(6,1); % Bar stiffness (N/m)
            stringDamping = env.cDamp*ones(24,1);  % String damping vector
            nodalMass = 3.2*ones(12,1); %1.625*ones(12,1);

            bar_radius = 0.025; % meters
            string_radius = 0.005;


            % Prepare environment
            env.nodes = [-barSpacing     barLength*0.5  0;
                     -barSpacing    -barLength*0.5  0;
                      barSpacing     barLength*0.5  0;
                      barSpacing    -barLength*0.5  0;
                      0             -barSpacing     barLength*0.5;
                      0             -barSpacing    -barLength*0.5;
                      0              barSpacing     barLength*0.5;
                      0              barSpacing    -barLength*0.5;        
                      barLength*0.5  0             -barSpacing;
                     -barLength*0.5  0             -barSpacing;
                      barLength*0.5  0              barSpacing;
                     -barLength*0.5  0              barSpacing];

            % Rotate superball to the "reset" position
            % 1/4 for z rotation (previous)
            HH=makehgtform('zrotate',9*pi/36);
            env.nodes = (HH(1:3,1:3)*env.nodes')';
            HH=makehgtform('xrotate',11*pi/36);
            env.nodes = (HH(1:3,1:3)*env.nodes')';

            env.nodes(:,3) = env.nodes(:,3) + 1*barLength;
            env.nodes(:,3) = env.nodes(:,3) - min(env.nodes(:,3)); % Make minimum node z=0 height.
            env.nodes(:,1) = env.nodes(:,1) - mean(env.nodes(:,1)); % Center x, y, axis
            env.nodes(:,2) = env.nodes(:,2) - mean(env.nodes(:,2));

            stringRestLength = 0.9*ones(24,1)*norm(env.nodes(1,:)-env.nodes(7,:));

            env.superBall = TensegrityStructure(env.nodes, env.strings, env.bars, zeros(12,3), env.stringstiffness,...
                barStiffness, stringDamping, nodalMass, env.delT, env.delT, stringRestLength,env.wallPos,env.wallNeg);

            env.superBallDynamicsPlot = TensegrityPlot(env.nodes, env.strings, env.bars, bar_radius, string_radius);
            
            %initialize rewards to 0
            env.rewards=0;
	    disp('Environment created');            
        end
        
        function f= createSuperBallGraph(env)
	    disp('start of function');
            barLength = 1.7;
            f = figure('units','normalized','outerposition',[0 0 1 1]);
            % use a method within TensegrityPlot class to generate a plot of the
            % structure
            generatePlot(env.superBallDynamicsPlot,gca);
            updatePlot(env.superBallDynamicsPlot,env.superBall.touchingWall);
	    disp('graph done');
            %settings to make it pretty
            axis equal
            view(3)
            grid on
            light('Position',[0 0 10],'Style','local')
            lighting flat
            colormap([0.8 0.8 1; 0 1 1]);
            lims = 1.2*barLength;
            xlim([-lims lims])
            ylim([-lims lims])
            zlim(1.6*[-0.01 lims])
            hold on

            % Draw Wall 1
            wall_x=[env.wallPos env.wallPos;env.wallPos env.wallPos];
            wall_y=[env.wallPos env.wallNeg;env.wallPos env.wallNeg];
            wall_z=[0 0;env.wallHeight env.wallHeight];

            walls=surf(wall_x,wall_y,wall_z);
            %Set wall 1 to be RED
            set(walls,'FaceColor',[1 0 0],'FaceAlpha',0.5);

            % Draw Wall 2
            wall_x=[env.wallNeg env.wallNeg;env.wallNeg env.wallNeg];
            wall_y=[env.wallPos env.wallNeg;env.wallPos env.wallNeg];
            wall_z=[0 0;env.wallHeight env.wallHeight];

            walls=surf(wall_x,wall_y,wall_z);
            %Set wall 2 to be GREEN
            set(walls,'FaceColor',[0 1 0],'FaceAlpha',0.5);

            % Draw Wall 3
            wall_x=[env.wallPos env.wallNeg;env.wallPos env.wallNeg];
            wall_y=[env.wallNeg env.wallNeg;env.wallNeg env.wallNeg];
            wall_z=[0 0;env.wallHeight env.wallHeight];

            walls=surf(wall_x,wall_y,wall_z);
            %Set wall 3 to be YELLOW
            set(walls,'FaceColor',[1 1 0],'FaceAlpha',0.5);

            % Draw Wall 4
            wall_x=[env.wallPos env.wallNeg;env.wallPos env.wallNeg];
            wall_y=[env.wallPos env.wallPos;env.wallPos env.wallPos];
            wall_z=[0 0;env.wallHeight env.wallHeight];

            walls=surf(wall_x,wall_y,wall_z);
            %Set wall 4 to be BLUE
            set(walls,'FaceColor',[0 0 1],'FaceAlpha',0.5);

            xlabel('x'); ylabel('y'); zlabel('z');
	    disp('function ends');
        end
        
        
        function observations=actionStep(env, actions)
            
            persistent tensStruct dynamicsPlot tspan 

            if nargin>1
                tensStruct = env.superBall;
                dynamicsPlot = env.superBallDynamicsPlot;
                tspan = env.tspan;
            end  
            

            %spoolingDistance=zeros(24,1);
            %motorsToMove=actions>0;
            for i=1:24
                %actions=1 means that rest length of string has to increase
                if actions(i)==1 
                    tensStruct.simStruct.stringRestLengths(i)= tensStruct.simStruct.stringRestLengths(i)+env.deltaSpool;
                    %spoolingDistance(i)=env.deltaSpool;
                %actions=2 means that rest length of string has to increase
                elseif actions(i)==2
                    tensStruct.simStruct.stringRestLengths(i)= tensStruct.simStruct.stringRestLengths(i)-env.deltaSpool;
                    %spoolingDistance(i)=-env.deltaSpool;
                end
                
                %Do not allow large or small string length measurements
%                 if actions(i)>0 
%                     tensStruct.simStruct.stringRestLengths(i) = newL;
%                 end
            end
            %Apply actions
            %tensStruct.simStruct.stringRestLengths(motorsToMove) = tensStruct.simStruct.stringRestLengths(motorsToMove)+spoolingDistance(motorsToMove);

            % Update nodes:
            dynamicsUpdate(tensStruct, tspan);
            dynamicsPlot.nodePoints = tensStruct.ySim(1:end/2,:);


            %Get new rest lengths after action is performed
            observations=tensStruct.simStruct.stringRestLengths;



        end
        
        %Function that returns the coordinates of the center of mass
        function coordinates=getCenterOfMass(env)
            coordinates=mean(env.superBallDynamicsPlot.nodePoints(:,3));
        end
               
        
        function updateGraph(env)
            
            updatePlot(env.superBallDynamicsPlot,env.superBall.touchingWall);

            drawnow  %plot it up
            
        end
        
        
        function reward= computeRewards(env)
%             if env.superBall.rewardTouchingGnd>0
              reward=env.superBall.rewardTouchingGnd;
%             else
%                 reward=0;
%             end
        end
        
        function done = computeDone(env)
            if env.rewards>30
                done=1;
            else
                done=0;
            end
        end
        
        
        %This function is used to reset the environment at the beginning of
        %every new episode
        function observations= envReset(env,render)
            
            resetSuperball(env,render);
            
            %wait for stabilization to be complete
            stabilizationUpdate(env,render);

            %reset rewards
            if render
                updatePlot(env.superBallDynamicsPlot,env.superBall.touchingWall);
                drawnow  %plot it up
            end
            env.rewards=0;
            
            %Reset observations
            observations=env.superBall.simStruct.stringRestLengths;
        end
        
        
        %This function lets superball bounce on the walls without any motor
        %being activated. It allows the robot to stabilize before starting
        %a learning episode
        function stabilizationUpdate(env,render)
            
            %Wait 50 cycles to make SuperBall stabilize before the new
            %episode
            i=0;
            
            while i<50
                % Update nodes:
                dynamicsUpdate(env.superBall, env.tspan);
                env.superBallDynamicsPlot.nodePoints = env.superBall.ySim(1:end/2,:);
                
                if render
                    updatePlot(env.superBallDynamicsPlot,env.superBall.touchingWall);
                    drawnow  %plot it up
                end
                i=i+1;
                
                if env.superBallDynamicsPlot.plotErrorFlag==1
                    %restart stabilization
                    disp('Restart Initialization');
                    i=0;
                    
                    resetSuperball(env,render);
                end
                
            end
            disp('Done Resetting');
            
        end
        
        %This function brings superBall in its initial position between the
        %walls
        function resetSuperball(env,render)
            
            env.stringstiffness = env.K*ones(24,1); % String stiffness (N/m)
            barStiffness = env.barStCoeff*ones(6,1); % Bar stiffness (N/m)
            stringDamping = env.cDamp*ones(24,1);  % String damping vector
            nodalMass = 1.625*ones(12,1);
            stringRestLength = 0.9*ones(24,1)*norm(env.nodes(1,:)-env.nodes(7,:));
            
            
            env.superBall = TensegrityStructure(env.nodes, env.strings, env.bars, zeros(12,3), env.stringstiffness,...
            barStiffness, stringDamping, nodalMass, env.delT, env.delT, stringRestLength,env.wallPos,env.wallNeg);
            
            if render
                updatePlot(env.superBallDynamicsPlot,env.superBall.touchingWall);
            end

        end
        
    end
    
end

