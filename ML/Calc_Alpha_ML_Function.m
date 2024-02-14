function [RawData] = Calc_Alpha_ML_Function(...
    Rtroc,F,vd_chemotaxis,ini_al,fin_al,al_step,deme_start,nl,Angle,Vo_max,xbias,Stage,DL)

%% Set up arrays to contain alpha values and average speeds for each value
% This program cycles through each deme to first give a range of alpha
% values to find which one is the best fit.  For each alpha value, 
% the function runs through 50 iterations and records the speed.
% Find_Alpha_Program uses these results to match which speed is closest to
% the theoretical value.  The closest value is matched to later be used to
% in curve fitting for the variable alpha value.

RawData = [];
for alpha = ini_al:al_step:fin_al
    
    %% The probability of Tumbling Up the Gradient
    d = 1.16; % Diffusion constant
    
    %% Choose Which Section of Model to Analyze
    
    pos = DL*deme_start; % Position on x-axis in µm
    pos_ini = DL*deme_start; % Initial position to calculate vd with
    % Calculate the position
    i = floor(pos/DL) + 1;
    theory_V = 0;
    deme_end = deme_start + 1;
    if deme_end > nl
        disp('Cannot use this value. Has to be smaller than nl.');
        return;
    end
    for i = deme_start:deme_end
        theory_V = theory_V + vd_chemotaxis(i);
    end
    Average_Theory_Vel = theory_V/(deme_end - deme_start + 1);
    
    %% Start the Time Loop
    Caculated_Ave_Vd_Array = zeros();
    iter = 1;
    dt = 0.1; % s
    while iter < 2000
        for t = 1:dt:1000 % time is in sec

            % Moving away from food.
            if (90 <= Angle) && (Angle < 270)
                Ptum = dt*exp(-d + alpha*Rtroc(i));
            else % Moving towards food.
                Ptum = dt*exp(-d - alpha*Rtroc(i));
            end

            R_rt = rand();
            if R_rt < Ptum
                % If r < P_tum the bacteria stays in the Tumble State
                [Next_Angle] = Tumble_Angle_Function(F);
                % Turning at Rotation = Next_Angle
                Angle = Angle + Next_Angle;
                Angle = round(Angle);
                if Angle >= 360
                    Angle = Angle - 360;
                end
            elseif R_rt >= Ptum
                % If r > P_run the bacteria stays in the Run State
                Dot_Product = cos(Angle*(pi/180));
                pos = pos + dt*Vo_max*Dot_Product;
            end

            %% CALCULATE WHERE THE POSITION IS HERE 
             % AND PREVENT IT FROM MOVING BEOND LIMIT

             if pos <= 0
                 pos = 0;
             elseif pos >= nl*DL
                 pos = nl*DL;
                 break
             end

            if i > deme_end
                break
            end
        end
        Caculated_Ave_Vd = (pos - pos_ini)/t;
        Caculated_Ave_Vd_Array(iter) = Caculated_Ave_Vd;
        pos = DL*deme_start; % Position is reset on x-axis in µm
        iter = iter + 1;
    end
        
    %% Record All the Data
    prob_tum_up = dt*exp(-d - alpha*Rtroc(deme_start)); % prob tumbling up
    prob_tum_down = dt*exp(-d + alpha*Rtroc(deme_start)); % prob tumbling down
    
    if (Stage > 2) % (mod(n,third_skip) == 0)
        % Fix values to store them into RawData as rows.
        position = deme_start*ones(iter - 1, 1);
        alpha_con = alpha*ones(iter - 1, 1);
        theoryVel = Average_Theory_Vel*ones(iter - 1, 1);
        prob_up = prob_tum_up*ones(iter - 1, 1);
        prob_down = prob_tum_down*ones(iter - 1, 1);
        % Append data for iteration to place into RawData.
        update = [position alpha_con theoryVel transpose(Caculated_Ave_Vd_Array) prob_up prob_down];
        RawData = [RawData; update];
    end
    % Update Position in Alpha_Array
end % for alpha = ini_al:al_step:fin_al

end