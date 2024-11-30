function Labels = TransformLabels(predicted)
TesLabels = predicted;
for i=1:length(TesLabels)
    if TesLabels(i) <= 70
        ActualL{i} = '1';
    elseif (TesLabels(i) > 70) && (TesLabels(i) < 130)
        ActualL{i} = '3';
    elseif TesLabels(i) >= 130
        ActualL{i} = '2';
    end
end

Labels = ActualL';
end

