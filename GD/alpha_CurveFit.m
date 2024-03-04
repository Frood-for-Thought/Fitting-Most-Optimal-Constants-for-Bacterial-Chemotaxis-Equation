Grad = 0.000405; % µm^-1
Max_Food_Conc = 60000; % µM

filename = 'closest_alpha_pos_1_to_101_MaxC_60000_Grad_0.000405.xlsx';
A = xlsread(filename);
format short g

X_data = A(:, 1);
Y_data = A(:, 2);

%% Fit: 'Polynomial_Fit'.

[xData, yData] = prepareCurveData( X_data, Y_data );

% Set up fittype and options.
ft = fittype( 'poly9' );

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft );

Results = [X_data, Y_data, fitresult(X_data)];
T_res = array2table(Results,'VariableNames',...
    {'X_data','Y_data','Curve_Fit'})

file_title = 'alpha_values_pos_%d_to_%d_MaxC_%d_Grad_%.6f_curve_fit.xlsx';
filename = sprintf(file_title, X_data(1), X_data(end), Max_Food_Conc,Grad);
writetable(T_res,filename,'Sheet',1,'Range','A1')



fitresult
gof

% Plot fit with data.
figure( 'Name', 'Polynomial_Fit' );
h = plot( fitresult, xData, yData );

% hold all
% % Label axes and legend
% m = plot(fitresult(X_data));
% legend( '-DynamicLegend', 'alpha values', 'initial polyfit', 'polynomial regression', 'Location', 'NorthEast' );

xlabel deme
ylabel alpha
xlim([1 101])
% grid on