result_predict_all = zeros(18,52);

sum_result_predict{1} = result_predict1 + result_predict2;

for i=1:18
    for j=1:52
        if sum_result_predict{1}(i,j) == 1 
            result_predict_all(i,j) = 1;
        end;
    end;
end;

sum_result_predict{2} = result_predict2 + result_predict3;

for i=1:18
    for j=1:52
        if sum_result_predict{2}(i,j) == 1 
            result_predict_all(i,j) = 2;
        end;
    end;
end;

sum_result_predict{3} = result_predict3 + result_predict4;

for i=1:18
    for j=1:52
        if sum_result_predict{3}(i,j) == 1 
            result_predict_all(i,j) = 3;
        end;
    end;
end;

sum_result_predict{4} = result_predict4 + result_predict5;

for i=1:18
    for j=1:52
        if sum_result_predict{4}(i,j) == 1 
            result_predict_all(i,j) = 4;
        end;
    end;
end;

for i=1:18
    for j=1:52
        if result_predict5(i,j) == 1 
            result_predict_all(i,j) = 5;
        end;
    end;
end;
