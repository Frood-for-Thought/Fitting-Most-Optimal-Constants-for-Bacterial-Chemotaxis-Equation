
function [Next_Angle] = Tumble_Angle_Function(F)
%% Radomly Select Tumble Angle of Rotation from Probability Distribution 
% P(x) = 0.5*(1+cos(x))*sin(x)

% n = 1;
% F = zeros();
% for x = 0:0.0175:pi
%     F(n) = 0.5*(1+cos(x))*sin(x);
%     n = n + 1;
% end

Prob_and_Angle = [F/sum(F); 1:180];
r_angle = rand();
sum_prob = 0;
for i = 1:length(Prob_and_Angle)
    sum_prob = sum_prob + Prob_and_Angle(1,i);
    if r_angle <= sum_prob
        Next_Angle = Prob_and_Angle(2,i);
        break
    end
end


% Prob_and_Angle = [F/sum(F); 1:180];
% Prob_and_Angle(:,1) = [];
% Prob_and_Angle(:,161:179) = [];
% P_Angle_new_location = randperm(length(Prob_and_Angle(2,:)));
% New_P_Angle_Order = Prob_and_Angle(:, P_Angle_new_location);
% r_angle = rand();
% Angle_Found = 0;
% while Angle_Found < 1
%     for i = 1:length(New_P_Angle_Order)
%         if r_angle < New_P_Angle_Order(1,i)
%             Location_Angle_Probability = New_P_Angle_Order(1,i);
%             Next_Angle = New_P_Angle_Order(2,i);
%             Angle_Found = 1;
%         end
%     end
%     r_angle = rand(); % if no angle is found
% end
