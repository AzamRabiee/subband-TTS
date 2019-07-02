clear all;
close all;
clc;

[wav_target, FS1] = audioread('./logdir/generate/4.200k.tar/01.wav');
[wav_output, FS2] = audioread('./logdir/generate/4.200k.out/01.wav');
load('./logdir/generate/4.200k/LJ001_0002.mat')

% plot subbands
for i=1:8
    subplot(4,2,i);
    plot(target(i, 1000:1300));
    hold on; plot(output(i, 1000:1300), 'r--')
    legend('target', 'predicted');
    ttl = sprintf('d%d', i);
    title(ttl);
end

figure;
title('fullband speech')
plot(wav_target(1000:1300));
hold on; plot(wav_output(1000:1300), 'r--')
legend('target', 'predicted');
