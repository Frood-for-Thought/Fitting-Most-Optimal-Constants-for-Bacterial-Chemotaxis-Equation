function [Record_Data_Array, iter, tot_time] = Calc_R_Per_lnr_Deme_Function(...
    F,ini_al,fin_al,al_step,deme_start,nl,Angle,Vo_max,DL)

%% Set up arrays to contain alpha values and average speeds for each value
% This program cycles through each deme to first give a range of alpha
% values to find which one is the best fit.  For each alpha value, 
% the function runs through iterations and records the distance travelled.

n = 1; % Initial row for Record_Data_Array
D = 112; % micrometres^2/s
Record_Data_Array = zeros();

for lnr = ini_al:al_step:fin_al
    %% The probability of Tumbling
    Pr_t = exp(-lnr);
    
    %% Choose Which Section of Model to Analyze
    
    pos = DL*deme_start; % Position on x-axis in µm
    pos_ini = DL*deme_start; % Initial position to calculate vd with
    dt = 0.1;
    %% Start the Time Loop
    iter = 1;
    % Go through several iterations to record all the data for this
    % diffusion constant
    while iter <= 30001
        % Go throught run-and-tumble algorithm
        tot_time = 1000;
        for t = 1:dt:tot_time % time is in sec
            % The rate of tumble for time "dt"
            Ptum = dt*Pr_t;

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
             end
        end
        %% Record All the Data
        Record_Data_Array(n,1) = D; % Theoretical Diff. Const.
        Record_Data_Array(n,2) = t; % Length of Time of Algorithm
        Record_Data_Array(n,3) = lnr; % Current Run/Tumble Diff. Const.
        Record_Data_Array(n,4) = 2*D*t; % Theoretical var = SD^2
        Record_Data_Array(n,5) = sqrt(2*D*t); % Theor. S.D. = SD^2
        Record_Data_Array(n,6) = pos_ini; % Initial starting position
        Record_Data_Array(n,7) = pos; % Final position
        Record_Data_Array(n,8) = pos_ini - pos; % Position Difference


        pos = DL*deme_start; % Position is reset on x-axis in µm
        iter = iter + 1; % Move on to the next iteration
        n = n + 1; % Move to the next row for Record_Data_Array
    end
end

end