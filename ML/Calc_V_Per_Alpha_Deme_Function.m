function [Record_Data_Array,Vel_Diff,skip,next_skip,third_skip] = Calc_V_Per_Alpha_Deme_Function(...
    Rtroc,F,vd_chemotaxis,ini_al,fin_al,al_step,deme_start,nl,Angle,Vo_max,xbias,Stage,DL)

%% Set up arrays to contain alpha values and average speeds for each value
% This program cycles through each deme to first give a range of alpha
% values to find which one is the best fit.  For each alpha value, 
% the function runs through iterations and records the speed.
% Find_Alpha_RawData_Program uses these results to match which speed is 
% closest to the theoretical value.

n = 1;
Record_Data_Array = zeros();
j = 1;
Vel_Diff = zeros();

for alpha = ini_al:al_step:fin_al
%     alpha
    
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

    Record_Data_Array(n,1) = alpha;
    Record_Data_Array(n,2) = deme_start;
    Record_Data_Array(n,3) = Average_Theory_Vel;
    Record_Data_Array(n,5) = Rtroc(deme_start)*alpha;
    dt = 0.1;
    
    % alpha*Rtroc needs to be greater than 0.01 to have any effect.
    if (Record_Data_Array(n,5) < 0.01) && (deme_start > 5)
        M = 0;
    else
        %% Start the Time Loop
        Caculated_Ave_Vd_Array = zeros();
        iter = 1;
        while iter < 100
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
                    % Turning at Rotation = Next_Angle/t_tum
                                    % Therefore, 
                    Angle = Angle + Next_Angle; %(Next_Angle*kt_r)*dt;
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
        M = mean(Caculated_Ave_Vd_Array); % Mean velocity for alpha
    end
    %% Record All the Data
    Record_Data_Array(n,4) = M;
    Record_Data_Array(n,6) = dt*exp(-d - alpha*Rtroc(deme_start)); % prob tumbling up
    Record_Data_Array(n,7) = dt*exp(-d + alpha*Rtroc(deme_start)); % prob tumbling down
    
    skip = 5;
    next_skip = 3;
    if (mod(n,skip) == 0) && (Stage < 2) % Stage 1 skip = 5
        AveVel_div_Skip = sum(Record_Data_Array((n-skip+1):n,4))/skip;
        Vel_Diff(j) = abs(AveVel_div_Skip - Average_Theory_Vel);
        if j > 2
            % In areas with low sensitivity, if velocity difference is not
            % consistently increasing then just skip this input.
            if (Vel_Diff(j) > Vel_Diff(j-1))&&(Record_Data_Array(n,5) > 0.1)
                break
            end
        end
        j = j + 1;
    elseif (Stage > 1) % Stage 2
        % Find a rough velocity difference between Mean run and the theoretical
        Vel_Diff(j) = abs(M - Average_Theory_Vel);
        j = j + 1;
    end
    
    % Update Position in Alpha_Array
    n = n + 1;
end % for alpha = ini_al:al_step:fin_al

end