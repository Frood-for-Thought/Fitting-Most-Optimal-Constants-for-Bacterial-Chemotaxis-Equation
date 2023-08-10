% FIND A WAY TO FIT THE ALPHA CONSTANT
% Find_Alpha_Program uses the results in "Calc_Vd_Per_Alpha_Deme_Function"
% to match which speed is closest to the theoretical value. 
% The closest value is matched to later be used to 
% in curve fitting to find the variable alpha value.

close all
%% INSEERT FOOD CONCENTRATION
% Gradient Variables
nl = 101;
Grad = 0.000405; % µm^-1
Max_Food_Conc = 60000; % µM
% Preset the function for the concentration of food.
Food_Function = zeros(nl,1);
DL = 310; % µm

for Food_Pos = 1:nl
    % x = Food_Pos*DL µm
    Food_Function(Food_Pos) = exp(Grad*Food_Pos*DL);
end
% Normalize with Food_Function(nl) to set maximum to the numerator value.
% Now the exponential function increases to Max_Food_Conc.
Ini_Food_Const = Max_Food_Conc/Food_Function(nl); % µM
% xbias describes the exponential concentration of food after
% normalization.
xbias = Ini_Food_Const*Food_Function;

%% Angle Probability Distribution
n = 1;
% The Angle Probability Distribution
F = zeros();
for x = 0:0.0175:pi
    F(n) = 0.5*(1+cos(x))*sin(x);
    n = n + 1;
end
Start_Angle = 90; % degrees
% plot(F)
% return
Angle = Start_Angle;

%% The chemotaxis speed matrix program

% Read the parameters from the excel file to use in the
% Time Rate of Change of the Receptor Bound equation.
file_title = 'input_parameters.xlsx';
A = xlsread(file_title);
vd_chemotaxis = transpose(A(:,2));
c_df_over_dc = transpose(A(:,3));
Vo_max = A(1,5);

clf
yyaxis left
plot(vd_chemotaxis,'b', 'LineWidth',2);
ylabel('Drift Velocity, Vd [µm/s]')
xlim([0 nl])
ylim([0 5])
yyaxis right
plot(xbias,'r')
ylabel('Concentration [µM]')
ylim([0 max(xbias)])
drawnow

%% Calculate the Time Rate of Change of the Fractional Amount of 
 % Receptor (Protein) Bound

Rtroc = zeros();
for i = 1:nl
    % The receptor time rate of change of protein bound, 
    % with respect to position.
    Rtroc(i) = vd_chemotaxis(i)*Grad*c_df_over_dc(i);
end

%% Choose the Deme Position

Pos_Alpha_Array = [];
for deme_start = 1:100
    %% Begin Calculation
    ini_al = 100;
    fin_al = 9000;
    al_step = 100;
    Stage = 1;
    [Record_Data_Array,Vel_Diff,skip,next_skip,third_skip] = Calc_V_Per_Alpha_Deme_Function(...
        Rtroc,F,vd_chemotaxis,ini_al,fin_al,al_step,deme_start,nl,Angle,Vo_max,xbias,Stage,DL);
    
    % Create a Table Out of Data Collected.
    T = array2table(Record_Data_Array,'VariableNames',{'Alpha','Deme','Theoretical_Velocity',...
        'Calculated_Velocity','AlphaXReceptor','Prob_Tumbling_Up',...
        'Prob_Tumbling_Down'})
    format short G
    % A Vel_Diff array was constructed to find how far the average velocity
    % calculated is from the theoretical. Pick the value closest to
    % theoretical velocity and use the location on the array to select 
    % Data_Row on Record_Data_Array to get an alpha value near where the
    % theoretical value should be.
    Vel_Diff
    [m,I] = min(Vel_Diff)
    Data_Row = 1 + (I-1)*skip;
    alpha_start = Record_Data_Array(Data_Row,1)

    %% Find the Next Best Alpha Value
    % Similar to the previous calculation but to further pinpoint a
    % range for use in stage 3.
    
    if alpha_start > 200
        ini_al = alpha_start - 200;
        fin_al = alpha_start + 800;
    else
        ini_al = 50;
        fin_al = alpha_start + 800;
    end
    al_step = 50;
    Stage = 2;
    [Record_Data_Array,Vel_Diff,skip,next_skip,third_skip] = Calc_V_Per_Alpha_Deme_Function(...
        Rtroc,F,vd_chemotaxis,ini_al,fin_al,al_step,deme_start,nl,Angle,Vo_max,xbias,Stage,DL);

    % Display the Function Data
    T = array2table(Record_Data_Array,'VariableNames',{'Alpha','Deme','Theoretical_Velocity',...
        'Calculated_Velocity','AlphaXReceptor','Prob_Tumbling_Up',...
        'Prob_Tumbling_Down'})
    format short G
    % A Vel_Diff array was constructed, pick the value closest to
    % theoretical velocity and use the location on the array to select 
    % Data_Row on Record_Data_Array to get another starting alpha value
    % even closer to the theoretical value.
    Vel_Diff
    [m,I] = min(Vel_Diff)
    Data_Row = 1 + (I-1)*next_skip
    alpha_start = Record_Data_Array(Data_Row,1)

    %% Given the range find the raw data for a list of alpha values.

    ini_al = alpha_start - 50;
    fin_al = alpha_start + 150;
    if vd_chemotaxis(deme_start) < 1
        al_step = 10;
    else
        al_step = 5;
    end
    
    % Now that the range is more determined, put this into a new file.
    % The file will generate a list of velocities over several iterations
    % for each alpha value and then output that list.
    % The data is then appended into Pos_Alpha_Array.
    Stage = 3;
    [RawData] = Calc_V_Per_Alpha_Deme_moreIter_Function(...
        Rtroc,F,vd_chemotaxis,ini_al,fin_al,al_step,deme_start,nl,Angle,Vo_max,xbias,Stage,DL);
    Pos_Alpha_Array = RawData;
    
%     Pos_Alpha_Array = [Pos_Alpha_Array; RawData];
%     T = array2table(Pos_Alpha_Array,...
%         'VariableNames',{'position','alpha','Theory_Vel', 'Calc_Velocity', 'Prob_Tum_Up', 'Prob_Tum_Down'})
%     format short G

    T = array2table(Pos_Alpha_Array,...
        'VariableNames',{'position','alpha','Theory_Vel', 'Calc_Velocity', 'Prob_Tum_Up', 'Prob_Tum_Down'});
    format short G

    file_title = 'pos_%d_alpha_values_MaxC_%d_Grad_%.6f.xlsx';
    filename = sprintf(file_title, deme_start,Max_Food_Conc,Grad);
    writetable(T,filename,'Sheet',1,'Range','A1')


%     if deme_start == 10
%         deme_start
%         vd_chemotaxis(deme_start)
%         break
%     end
end

% T = array2table(Pos_Alpha_Array,...
%     'VariableNames',{'position','alpha','Theory_Vel', 'Calc_Velocity', 'Prob_Tum_Up', 'Prob_Tum_Down'});
% format short G
% 
% file_title = 'pos_%d_alpha_%d_to_%d_MaxC_%d_Grad_%.6f.xlsx';
% filename = sprintf(file_title, deme_start,ini_al,fin_al,Max_Food_Conc,Grad);
% writetable(T,filename,'Sheet',1,'Range','A1')
