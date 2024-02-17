function [Pos_Alpha_Array] = Calc_Alpha_ML_Function(...
    Rtroc,F,vd_chemotaxis,alpha_start,deme_start,nl,Angle,Vo_max,xbias,DL)

%% Set up arrays to contain alpha values and average speeds for each value
alpha = alpha_start

%% The probability of Tumbling Up the Gradient
d = 1.16; % Diffusion constant
dt = 0.1; % s time step
loss = NaN;
loss_final = 100;
alpha_final = 0;

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

%% Start the ML Loop

for n = 1:90 % start training loop
    M = 0;
    Calculated_Ave_Vd_Array = zeros();
    if n <= 20
        max_iter = 1000;
        TV = 1/(100*Rtroc(deme_start));
    elseif (n > 20) && (n <= 40)
        max_iter = 2000;
        TV = 1/(200*Rtroc(deme_start));
    elseif (n > 40) && (n < 60)
        max_iter = 4000;
        TV = 1/(300*Rtroc(deme_start));
    else
        max_iter = 10000;
        TV = 1/(300*Rtroc(deme_start));
    end
    iter = 1;
    while iter < max_iter
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
        Calculated_Ave_Vd = (pos - pos_ini)/t;
        Calculated_Ave_Vd_Array(iter) = Calculated_Ave_Vd;
        pos = DL*deme_start; % Position is reset on x-axis in µm
        iter = iter + 1;
    end

    M = mean(Calculated_Ave_Vd_Array); % Mean velocity for alpha
    n
    Vel_Diff = M - Average_Theory_Vel;
    stderror = std(Calculated_Ave_Vd_Array) / sqrt(length(Calculated_Ave_Vd_Array));
    
    if n > 60
        loss = [(2*TV*Vel_Diff)^2]
    end
    if (loss < loss_final) && (n > 60) && (abs(TV_2_SE) > abs(h))
        alpha_final = alpha
        loss_final = loss
    end
    
    h = - 2*TV*Vel_Diff
    TV_2_SE = 2*TV*stderror
    
    alpha = alpha + h
    if alpha < 0
        alpha = 1;
    end

end % for n = ... end training loop

alpha_final
loss_final

return
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
    Pos_Alpha_Array = [Pos_Alpha_Array; update];
end

end