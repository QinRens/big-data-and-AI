figure
direction = data(2:end)-data(1:end-1)
plot(direction)
direction=direction>0
csvwrite('baiduDirection.csv',direction)





%%
loss_vec = csvread('loss_vec.csv',1);
figure
plot(loss_vec)
xlabel('Iteration')
ylabel('Loss')
test_acc = csvread('test_acc.csv',1);
train_acc = csvread('train_acc.csv',1);
figure
plot(test_acc)
hold on
plot(train_acc)
legend('test','train')
xlabel('Iteration')
ylabel('Accuracy')
title('Baidu stocks direction prediction(logistic regression)')
%%
figure
[n,xout]=hist(UD500, 10);
bar(xout,n/length(UD500));

hold on
ylabel('Frequency')
title('S&P500 difference data frequency histogram ')

