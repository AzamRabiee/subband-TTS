clear all;
close all;
clc;

load('./logdir/train/6/tar_out_60000.mat')

plot(target(8, 1000:1300));
hold on; plot(predicted1(8, 1000:1300), 'k')
hold on; plot(predicted2(8, 1000:1300), 'r--')
legend('target', 'predicted 1', 'predicted 2');

figure; 
subplot(211);
title('target')
target = reshape(target_256, [19935, 2048]);
imagesc(target'); xlabel('time'); ylabel('subbands (every 256)');
colorbar
subplot(212);
title('predicted 1')
predicted1_256 = reshape(predicted1_256, [19935, 2048]);
imagesc(predicted1_256'); xlabel('time'); ylabel('subbands (every 256)');
colorbar