function wavelet_recons()
    clear all;
    close all;
    FS = 16000;

%    %{
    % option 1: one wave file
    out_dir = './logdir/generate/4';
    mkdir(out_dir);

    variables = load('./logdir/generate/4/LJ001_0002_threading.mat');
    x = wav_reconstruction(variables.target);
%     x_est = wav_reconstruction(variables.output);
    x_est1 = wav_reconstruction(variables.predicted1);
    x_est1 = x_est1(end-length(x)+1: end);

    audiowrite(strcat(out_dir, '/tar_200k_threading.wav'), x, FS);
    audiowrite(strcat(out_dir, '/out1_200k_threading.wav'), x_est1, FS);
    %}
    
    % option 2: multiple wave files in a folder
%     out_dir_tar = './logdir/generate/4.200k.tar';
%     mkdir(out_dir_tar);
%     out_dir_gen = './logdir/generate/4.200k.out';
%     mkdir(out_dir_gen);
%     mat_files = dir('./logdir/generate/4.200k/*.mat');
%     for i=1:numel(mat_files)
%         variables = load([mat_files(i).folder '/' mat_files(i).name]);
%         x = wav_reconstruction(variables.target);
%         x_est = wav_reconstruction(variables.output);
%         x_est = x_est(end-length(x)+1: end);
%         tar_path = [out_dir_tar, '/' mat_files(i).name(9:end-4) '.wav'];
%         out_path = [out_dir_gen, '/' mat_files(i).name(9:end-4) '.wav'];
%         audiowrite(tar_path, x, FS);
%         audiowrite(out_path, x_est, FS);
%     end

end

function wav = wav_reconstruction(coeff)
    n_levels = 8;
    wname = 'db10';
    coeff(n_levels + 1,:) = 0;
    wav = iswt(coeff, wname);
end

%{
% plot target, master, subband
master_tar = '../wavenet_tts_master/logdir/generate/1.repeat_mono/target.wav';
master_out = '../wavenet_tts_master/logdir/generate/1.repeat_mono/LJ001-0002-phone.npy_gen.wav';
[x_master_tar, FS] = audioread(master_tar);
[x_master_out, FS] = audioread(master_out);

x_master_out = x_master_out(end-length(x_master_tar)+1: end);
x = x(end-length(x_master_tar) + 1:end);
x_est = x_est(end-length(x_master_tar) + 1:end);

s=11200;
e=11600;
subplot(211); plot(s:e, x_master_tar(s:e)); hold on; plot(s:e, x_master_out(s:e), 'r--');
title('Wavenet master'); xlim([s e]);
legend('target','output');
subplot(212); plot(s:e, x(s:e)); hold on; plot(s:e, x_est(s:e), 'r--')
title('Wavenet subband');
legend('target','output');xlim([s e]);

figure; plot(s:e, x_master_tar(s:e), '--', 'LineWidth', 2); 
hold on; plot(s:e, x_master_out(s:e), 'LineWidth', 2);
hold on; plot(s:e, x_est(s:e), 'LineWidth', 2);
title('Wavenet generated output'); xlim([s e]);
legend('target','master', 'subband');
%}
