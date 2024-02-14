function [L] = Loss_Function(...
    Average_Theory_Vel, Calculated_Ave_Vd_Array, alpha)
    L = 0;
    for i = 1:length(length(Calculated_Ave_Vd_Array))
        L = L + (Calculated_Ave_Vd_Array(i) - Average_Theory_Vel)^2;
    end
    L = L/length(Calculated_Ave_Vd_Array);
end