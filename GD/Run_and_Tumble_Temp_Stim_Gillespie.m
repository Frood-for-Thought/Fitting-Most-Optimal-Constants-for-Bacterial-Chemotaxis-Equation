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

for Food_Pos = 1:nl
    % x = Food_Pos*300 痠
    Food_Function(Food_Pos) = exp(Grad*Food_Pos*300);
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
Average_Diff = sum(xbias)/((nl-(Non0_Deme-1))*300); % [然/痠]
Difference_Label = 'The average difference in concentration between all demes with attractant is %.2f 然/痠';
Initial_Conc_Label = 'The initial concentration at deme %d is %.2f 然';
Final_Conc_Label = 'The largest concentration at deme %d is %.2f 然';
A = sprintf(Initial_Conc_Label, Non0_Deme, xbias(Non0_Deme))
B = sprintf(Final_Conc_Label, nl, xbias(nl))
C = sprintf(Difference_Label,Average_Diff)

%% The chemotaxis speed matrix program
[vd_chemotaxis,c_df_over_dc, Vo_max] = Attractant_Mig_Function(nl,Grad,xbias,Max_Food_Conc);

f1 = figure;
f2 = figure;
f3 = figure;
f4 = figure;
f5 = figure;

% Plot the c*dF/dc wrt Position

% yyaxis left
% plot(c_df_over_dc,'b', 'LineWidth',2);
% ylabel('C*dF/dC, [Unitless]')
% xlim([0 nl])
% ylim([0 2.5])
% yyaxis right
% plot(xbias,'r')
% ylabel('Concentration [然]')
% ylim([0 max(xbias)])
% return

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
file_title = 'alpha_MaxC_%d_Grad_%.6f.csv';
filename = sprintf(file_title,Max_Food_Conc,Grad);
A = xlsread(filename);
format short g
alpha = transpose(A(:,4));
% alpha = A.*0;
lnr = 1.16; % Nice
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

% Plot Prob of Tumble and c*dF/dc wrt Position vs Concentration
% figure(f3);
% clf
% yyaxis left
% plot(Pr_t,'b', 'LineWidth',2);
% ylabel('Probability of Tumble (blue) and C*dF/dC (green)')
% xlim([0 nl])
% hold on
% plot(c_df_over_dc,'g', 'LineWidth',2);
% yyaxis right
% plot(xbias,'r')
% ylabel('Concentration [然]')
% ylim([0 max(xbias)])
% return

figure(f4);
clf
yyaxis left
plot(Pr_t_up,'b', 'LineWidth',2);
ylabel('Rate of Tumble Up (blue) and Down (green) Gradient [s^-1]')
xlim([0 nl])
hold on
plot(Pr_t_down,'g', 'LineWidth',2);
yyaxis right
plot(xbias,'r')
ylabel('Concentration [然]')
ylim([0 max(xbias)+0.1])

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

i = 40;
ini_deme = i;
pos = 300*i - 300; % Position on x-axis in 痠
pos_ini = pos; % Initial position to calculate vd with
% Calculate the position
i = floor(pos/300) + 1;
% Values used for calculating average tumble rates up and down gradient
Sum_Tumble_Rate_Up = 0;
Sum_Tumble_Rate_Down = 0;
iter = 0;
% Initializing time
dt = 0;
t = 0;
Max_time = 1000;
% In this Gillespie could only calculate the rate of a tumble
while t < Max_time % time is in sec

%     kr_t = kCCW_CW;
%     kt_r = kCW_CCW;

    % If bacteria is moving down the gradient
    if (90 <= Angle) && (Angle < 270)
        R_tum = Pr_t_down(i);
        Sum_Tumble_Rate_Down = Sum_Tumble_Rate_Down + R_tum;
    % Bacteria is moving up the gradient
    else
        R_tum = Pr_t_up(i);
        Sum_Tumble_Rate_Up = Sum_Tumble_Rate_Up + R_tum;
    end
    
    % After calculating the rate of all the reactions,
    % the time progression should be updated for just the specific
    % tumble reaction at that deme so as to make sure that the bacteria's
    % movement speed is accurate.  As well, the other rates of reaction
    % are assumed to occur when the bacteria is mostly stationary.
    R_tot = R_tum;
    
%% TIME PROGRESSION
    r1 = rand();
    dt = (1/R_tot)*log(1/r1);
    t = t + dt; % new time
    if t > Max_time
        break
    end
%% RANDOMLY SELECT RATE
    % In this case there is only one rate reaction, the bacteria tumbling
    R_rt = rand();
        % The bacteria moves in the alloted time dt
        Dot_Product = cos(Angle*(pi/180));
        pos = pos + dt*Vo_max*Dot_Product;
    
        % The bacteria tumbles and reorients with an updated position,
        % Angle + Next_Angle, taken from the function 
        % Tumble_Angle_Function using a probability distribution, F
        [Next_Angle] = Tumble_Angle_Function(F);
        Angle = Angle + Next_Angle;
        Angle = round(Angle);
        if Angle >= 360
            Angle = Angle - 360;
        end
    
    Angle
    pos
    
    % Figure with Phasor and Position
    figure(f5);
    clf
    if abs(mod(t,10)) == 0
        x = cos(Angle*(pi/180));
        y = sin(Angle*(pi/180));
        subplot(2,1,1)
        compass(x,y)
        title('Angle Bacteria is Facing [^o]')
        
        subplot(2,1,2)
        plot(pos,0,'o')
%         xlim([-10 10])
        line(xlim, [0 0])
        title('Position Along Gradient [痠]')
        drawnow
    end
    
    %% CALCULATE WHERE THE POSITION IS HERE 
     % AND PREVENT IT FROM MOVING BEOND LIMIT
     
     if pos <= 0
         pos = 300*i - 300;
     elseif pos >= nl*300
         pos = nl*300;
         break
     end
    
    % Calculate the position
    i = floor(pos/300) + 1;
    if i == nl
        Tumble_Dir = 1;
    elseif xbias(i+1) > xbias(i)
        Tumble_Dir = 1;
    else
        Tumble_Dir = -1;
    end
    i
    t
    iter = iter + 1;
end
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

