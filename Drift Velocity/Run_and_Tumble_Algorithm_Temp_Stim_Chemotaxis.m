close all
%% INSEERT FOOD CONCENTRATION
% This program reproduces the Temporal Stimulation Migration but is now
% reworked to include the fractional amount of receptor bound using the
% receptor sensitivity work from Sourjik at al and Yu Hai Tu et al.

%% Gradient Variables
nl = 101;
Grad = 0.000405; % 痠^-1
Max_Food_Conc = 60000; % 然
Food_Function = zeros(nl,1);
DL = 310; % The deme length (痠)

for Food_Pos = 1:nl
    Food_Function(Food_Pos) = exp(Grad*Food_Pos*DL);
end
% Normalize with Food_Function(nl) to set Maximum to the numerator value
% Food_Function = Food_Function/Food_Function(nl); %Normalize Food Function.
Ini_Food_Const = Max_Food_Conc/Food_Function(nl); % 然
xbias = Ini_Food_Const*Food_Function;

Mark_Ini_Non_Zero_Deme = 0;
for i = 1:nl
    if (xbias(i) > 0.009) && (Mark_Ini_Non_Zero_Deme < 1)
        Non0_Deme = i;
        Mark_Ini_Non_Zero_Deme = 1;
    end
end
Average_Diff = sum(xbias)/((nl-(Non0_Deme-1))*DL); % [然/痠]
Difference_Label = 'The average difference in concentration between all demes with attractant is %.2f 然/痠';
Initial_Conc_Label = 'The initial concentration at deme %d is %.2f 然';
Final_Conc_Label = 'The largest concentration at deme %d is %.2f 然';
A = sprintf(Initial_Conc_Label, Non0_Deme, xbias(Non0_Deme))
B = sprintf(Final_Conc_Label, nl, xbias(nl))
C = sprintf(Difference_Label,Average_Diff)

%% The chemotaxis speed matrix program
% Read the parameters from the excel file to use in the
% Time Rate of Change of the Receptor Bound equation.
file_title = 'input_parameters.xlsx';
A = xlsread(file_title);
vd_chemotaxis = transpose(A(:,2));
c_df_over_dc = transpose(A(:,3));
Vo_max = A(1,5);

f1 = figure;
f2 = figure;
f3 = figure;
f4 = figure;
f5 = figure;

% Plot the Chemotaxis Migration Speed
figure(f1);
clf
yyaxis left
plot(vd_chemotaxis,'b', 'LineWidth',2);
ylabel('Theoretical Drift Velocity, Vd [痠/s]')
xlim([0 nl])
ylim([0 5])
yyaxis right
plot(xbias,'r')
ylabel('Concentration [然]')
ylim([0 max(xbias)])

%% Calculate the Time Rate of Change of the Fractional Amount of 
 % Receptor (Protein) Bound

Rtroc = zeros();
for i = 1:nl
    % The receptor time rate of change, with respect to position
    Rtroc(i) = vd_chemotaxis(i)*Grad*c_df_over_dc(i);
end

% Plot the Receptor time rate of change bount wrt position
figure(f2);
clf
yyaxis left
plot(Rtroc,'b', 'LineWidth',2);
ylabel('Time Rate of Change for Receptor Protein Bound, [1/s]')
xlim([0 nl])
yyaxis right
plot(xbias,'r')
ylabel('Concentration [然]')
ylim([0 max(xbias)])

%% Open the Alpha Values

Pr_t_up = zeros();
Pr_t_down = zeros();
file_title = 'alpha_values_MaxC_60000_Grad_0.000405_curve_fit.csv';
filename = sprintf(file_title,Max_Food_Conc,Grad);
A = xlsread(filename);
format short g
alpha = transpose(A(:,4));
lnr = 1.16;
for i = 1:nl
    % The probability of tumbling up and down
    Pr_t_up(i) = exp(-lnr - alpha(i)*Rtroc(i));
    Pr_t_down(i) = exp(-lnr + alpha(i)*Rtroc(i));
end

figure(f3);
clf
plot(alpha,'r.')
ylabel('Alpha Value [s^-1]')
xlim([0 (nl+1)])

figure(f4);
clf
yyaxis left
plot(Pr_t_up,'b', 'LineWidth',2);
ylabel('Probability of Tumble Up (blue) and Down (green) Gradient')
xlim([0 nl])
hold on
plot(Pr_t_down,'g', 'LineWidth',2);
yyaxis right
plot(xbias,'r')
ylabel('Concentration [然]')
ylim([0 max(xbias)])

%% Angle Probability Distribution
n = 1;
% The Angle Probability Distribution
F = zeros();
for x = 0:0.0175:pi
    F(n) = 0.5*(1+cos(x))*sin(x);
    n = n + 1;
end
Start_Angle = round(360*rand());
Angle = Start_Angle;

%% Run and Tumble Algorithm

i = 1;
ini_deme = i;
pos = DL*i - DL; % Position on x-axis in 痠
pos_ini = pos; % Initial position to calculate vd with
% Calculate the position
i = floor(pos/DL) + 1;
% Values used for calculating average tumble rates up and down gradient
Sum_Tumble_Rate_Up = 0;
Sum_Tumble_Rate_Down = 0;
iter = 0;
% Initializing Time
dt = 0.1;
record_time = [];
record_pos = [];

% Open video file to record
v = VideoWriter('BacRTMig.avi');
open(v)
video_Condition = 0;

for t = 1:dt:25000 % time is in sec
    video_Condition = 1;
%     kr_t = kCCW_CW;
%     kt_r = kCW_CCW;

    % If bacteria is moving down the gradient
    if (90 <= Angle) && (Angle < 270)
        Ptum = dt*Pr_t_down(i);
    % Bacteria is moving up the gradient
    else
        Ptum = dt*Pr_t_up(i);
    end
    
    R_rt = rand();
    if R_rt < Ptum
        % If rand < P_tum the bacteria stays in the Tumble State
        [Next_Angle] = Tumble_Angle_Function(F);
        % Turning at Rotation = Next_Angle/t_tum
                        % Therefore, 
        Angle = Angle + Next_Angle; %(Next_Angle*kt_r)*dt;
        Angle = round(Angle);
        if Angle >= 360
            Angle = Angle - 360;
        end
    elseif R_rt >= Ptum
        % If rand > P_run the bacteria stays in the Run State
        Dot_Product = cos(Angle*(pi/180));
        pos = pos + dt*Vo_max*Dot_Product;
    end

    % Figure with Phasor and Position
    figure(f5);
    clf
    if abs(mod(t,0.5)) == 0
        % Set the size of the figure
        f5.Position = [300 100 800 840];
        
        x = cos(Angle*(pi/180));
        y = sin(Angle*(pi/180));
        subplot(2,1,1)
        compass(x,y)
        title('Angle Bacteria is Facing [^o]')
        Time_In_Min = t/60;
        xlabel(['Time is ',num2str(Time_In_Min,'%4.2f'),' min'], 'fontsize', 12)
        
        subplot(2,1,2)
        yyaxis left
        plot(i,vd_chemotaxis(i),'o', 'LineWidth', 2, 'Color', 'm')
        hold on
        plot(vd_chemotaxis,'b', 'LineWidth',2)
        xlim([1 nl])
        line(xlim, [0 0])
        title('Position Along Gradient [axis i = 310 痠]')
        ylabel ({'Theoretical Drift Vel. [痠/s] (blue line)';'Actual R-T Position [痠] (purple circle)'}, 'fontsize', 13)
        X_lab = sprintf('Position = %.2f 痠', pos);
        xlabel (X_lab);
            yyaxis right
            plot(xbias,'r')
            ylabel('Attractant Concentration [然]')
            ylim([0 max(xbias)])
        
%         if mod(t, 5) == 0
            frame = getframe(gcf);
            writeVideo(v,frame)
%         end

        drawnow;
        video_Condition = 0;
        
    end
    
    %% CALCULATE WHERE THE POSITION IS HERE 
     % AND PREVENT IT FROM MOVING BEOND LIMIT
     
     if pos <= 0
         pos = DL*i - DL;
     elseif pos >= nl*DL
         pos = nl*DL;
         break
     end
    
    % Calculate the position
    i = floor(pos/DL) + 1;
    if i == nl
        Tumble_Dir = 1;
    elseif xbias(i+1) > xbias(i)
        Tumble_Dir = 1;
    else
        Tumble_Dir = -1;
    end
    
    i
    pos
    t
    % Get a recording of the time and position
    if abs(mod(t,1)) == 0
        record_time = [record_time; t];
        record_pos = [record_pos; pos];
    end
    
    if (90 <= Angle) && (Angle < 270)
        Sum_Tumble_Rate_Down = Sum_Tumble_Rate_Down + Ptum;
    else
        Sum_Tumble_Rate_Up = Sum_Tumble_Rate_Up + Ptum;
    end
    iter = iter + 1;
end

close(v);

% Put the recorded position and time in a file.
T = array2table([record_time record_pos],'VariableNames',...
    {'time','position'})

file_title = 'pos_v_time_MaxC_%d_Grad_%.6f.xlsx';
filename = sprintf(file_title, Max_Food_Conc,Grad);
writetable(T,filename,'Sheet',1,'Range','A1')

final_deme = i;
time = t;
vd = (pos - pos_ini)/t;
Tumble_Rate_Up = Sum_Tumble_Rate_Up/iter;
Tumble_Rate_Down = Sum_Tumble_Rate_Down/iter;
Results = zeros();
Results(1,1) = vd;
Results(1,2) = Tumble_Rate_Up;
Results(1,3) = Tumble_Rate_Down;
Results(1,4) = ini_deme;
Results(1,5) = final_deme;
Results(1,6) = time/60;
Results(1,7) = Grad;
Results(1,8) = Max_Food_Conc;
Results(1,9) = Vo_max;

T_res = array2table(Results,'VariableNames',...
    {'Drift_Velocity','Ave_Rtum_Up','Ave_Rtum_Down','Initial_Deme','Final_Deme','Time_in_min',...
    'Gradient','Max_Food_Concentration', 'Max_Speed'})

