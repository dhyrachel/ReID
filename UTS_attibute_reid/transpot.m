Hist_query = importdata('Hist_query.mat');
Hist_test = importdata('Hist_test.mat');

Hist_test = Hist_test'
Hist_query = Hist_query'

save('Hist_query.mat', 'Hist_query');
save('Hist_test.mat', 'Hist_test');
