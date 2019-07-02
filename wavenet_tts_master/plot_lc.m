clear all;
close all;
% load('/home/cnsl/TTS/wavenet_tts/wavenet_tts_master/logdir/generate/20.avg_3conv_repeat/LJ001-0002.mat')
load('/home/cnsl/TTS/wavenet_tts/wavenet_tts_master/logdir/generate/24.100k/LJ001-0003.mat')

% h: phoneme seq according to the wave time samples
% h_hat: averaged h, every 256 samples (16ms) 
% a3: activation of the third conv. layer
% C: upsampled by repeat
l=size(h1, 2);

subplot(515);
imagesc(h'); 
title('phoneme sequence according to the wave time samples'); 
xlabel('wave time index'); ylabel('phoneme index'); colorbar

subplot(514);
imagesc(h_hat'); 
title('averaged phoneme sequence, every 256 samples (16 ms)'); 
xlabel('sequence index'); ylabel('phoneme index')
xlim([1 l]); colorbar

subplot(513);
imagesc(squeeze(h1)'); 
title('activation of the first conv. layer'); 
xlabel('sequence index'); ylabel('channel index')
xlim([1 l]); colorbar

subplot(512);
imagesc(squeeze(h2)'); 
title('activation of the second conv. layer'); 
xlabel('sequence index'); ylabel('channel index')
xlim([1 l]); colorbar

subplot(511);
imagesc(squeeze(h3)'); 
title('activation of the last conv. layer'); 
xlabel('sequence index'); ylabel('channel index')
xlim([1 l]); colorbar
% 
% subplot(611);
% imagesc(C'); 
% title('upsampled activation as the conditioning feature'); 
% xlabel('wave time index'); ylabel('channel index'); colorbar
