function Accuracy = findaccuracy(predicted,Actual)
acc_count = strcmp(predicted,Actual); 
Accuracy = nnz(acc_count)/length(acc_count);
end

