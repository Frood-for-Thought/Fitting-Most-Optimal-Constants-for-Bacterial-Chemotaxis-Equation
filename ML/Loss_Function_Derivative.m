function [dL] = Loss_Function_Derivative(...
    Average_Theory_Vel, Calculated_Ave_Vd_Array)
    dL = 0;
    for i = 1:length(length(Calculated_Ave_Vd_Array))
        dL = dL + (Calculated_Ave_Vd_Array(i) - Average_Theory_Vel);
    end
    dL = (2/length(Calculated_Ave_Vd_Array))*dL;
end