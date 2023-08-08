% FIND A WAY TO FIT THE DIFF CONSTANT
% Find_Diff_Program uses the results in "Calc_R_Per_lnr_Deme_Function"
% to match which run and tumble is closest to the theoretical value. 
% The closest value is matched to later be used in curve fitting to 
% find the run and tumble diffusion constant.

close all
DL = 310; % Deme Length in micrometres
nl = 101; % The number of Demes
Vo_max = 16/sqrt(2); % Run speed

%% Angle Probability Distribution
n = 1;
% The Angle Probability Distribution
F = zeros();
for x = 0:0.0175:pi
    F(n) = 0.5*(1+cos(x))*sin(x);
    n = n + 1;
end
Start_Angle = 0;
Angle = Start_Angle;

%% Choose the Deme Position
% deme_start = 10;
Pos_Alpha_Array = zeros();
deme_start = 50;

    %% Begin Calculation
    ini_al = 1.00; % The initial diffusion constant to test
    fin_al = 1.01; % The final diffusion constant to test
    al_step = 0.01; % The increment step
    Stage = 1;
    [Record_Data_Array, iter, tot_time] = Calc_D_Per_r_Deme_Function(...
    F,ini_al,fin_al,al_step,deme_start,nl,Angle,Vo_max,DL);
    
    % Create a Table Out of Data Collected
    T = array2table(Record_Data_Array,'VariableNames',{'D','Time','R_T_Diff_Con',...
        'Std_Dev','Var','Ini_Pos','Final_Pos','Pos_Difference'})
    format short G

file_title = '%.3f_to_%.3f_Diff_Conts_%.f_DataPts_%.f_Time.xlsx';
filename = sprintf(file_title,ini_al,fin_al,iter,tot_time)
writetable(T,filename,'Sheet',1,'Range','A1')
