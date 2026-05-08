% Generate official COST2100 datasets for Exercise 2.15 and export them in
% the CsiNet-compatible .mat layout used by scripts/run_exercise_2_15_tf.py.
%
% Manual execution example in MATLAB:
%
%   cd('D:\NYCU\class\Artificial Intelligence Wireless\NYCU-AI-Wireless-Communication-HW\Exercise_2_15_ MidTerm')
%   generate_cost2100_csinet_datasets('D:\NYCU\class\Artificial Intelligence Wireless\cost2100')
%
% Notes:
% - This script is intentionally conservative because the public COST2100
%   MATLAB API differs slightly across revisions. If your local COST2100
%   checkout returns a different channel field, adjust cost2100_frequency_response().
% - The exported files match the Python pipeline:
%     data/cost2100_official/<dataset>/DATA_Htrain.mat       key: HT
%     data/cost2100_official/<dataset>/DATA_Hval.mat         key: HT
%     data/cost2100_official/<dataset>/DATA_Htest.mat        key: HT
%     data/cost2100_official/<dataset>/DATA_HtestF_all.mat   key: HF_all

function generate_cost2100_csinet_datasets(cost2100_root, output_root, samples_per_split)
    if nargin < 1 || isempty(cost2100_root)
        cost2100_root = fullfile(pwd, '..', 'cost2100');
    end
    if nargin < 2 || isempty(output_root)
        output_root = fullfile(pwd, 'data', 'cost2100_official');
    end
    if nargin < 3 || isempty(samples_per_split)
        samples_per_split = struct('train', 1200, 'val', 300, 'test', 400);
    end

    addpath(genpath(fullfile(cost2100_root, 'matlab')));
    if ~exist(output_root, 'dir')
        mkdir(output_root);
    end

    dataset_specs = {
        cost2100_spec('D1_indoor_uniform',    'IndoorHall_5GHz', 20,  'uniform')
        cost2100_spec('D2_indoor_center',     'IndoorHall_5GHz', 20,  'center')
        cost2100_spec('D3_indoor_edge',       'IndoorHall_5GHz', 20,  'edge')
        cost2100_spec('D4_indoor_ring',       'IndoorHall_5GHz', 20,  'ring')
        cost2100_spec('D5_outdoor_uniform',   'SemiUrban_300MHz', 400, 'uniform')
        cost2100_spec('D6_outdoor_clustered', 'SemiUrban_300MHz', 400, 'clustered')
    };

    rng(535100);
    for idx = 1:numel(dataset_specs)
        spec = dataset_specs{idx};
        fprintf('Generating %s...\n', spec.name);
        dataset_dir = fullfile(output_root, spec.name);
        if ~exist(dataset_dir, 'dir')
            mkdir(dataset_dir);
        end

        [HT, ~] = generate_split(spec, samples_per_split.train);
        save(fullfile(dataset_dir, 'DATA_Htrain.mat'), 'HT', '-v7');

        [HT, ~] = generate_split(spec, samples_per_split.val);
        save(fullfile(dataset_dir, 'DATA_Hval.mat'), 'HT', '-v7');

        [HT, HF_all] = generate_split(spec, samples_per_split.test);
        save(fullfile(dataset_dir, 'DATA_Htest.mat'), 'HT', '-v7');
        save(fullfile(dataset_dir, 'DATA_HtestF_all.mat'), 'HF_all', '-v7');
    end
end

function spec = cost2100_spec(name, scenario, area_length_m, user_distribution)
    spec = struct();
    spec.name = name;
    spec.scenario = scenario;
    spec.area_length_m = area_length_m;
    spec.user_distribution = user_distribution;
    spec.nt = 32;
    spec.subcarriers = 1024;
    spec.keep_delay_rows = 32;
    spec.keep_frequency_bins = 125;
end

function [HT, HF_all] = generate_split(spec, sample_count)
    HT = zeros(sample_count, 32 * 32 * 2, 'single');
    HF_all = complex(zeros(sample_count, 32, 125, 'single'));

    for sample_idx = 1:sample_count
        success = false;
        for attempt = 1:30
            try
                user_pos = sample_user_position(spec);
                H_freq = cost2100_frequency_response(spec, user_pos);
                success = true;
                break;
            catch err
                if attempt == 30
                    rethrow(err);
                end
            end
        end
        if ~success
            error('Unable to generate sample %d for %s.', sample_idx, spec.name);
        end
        [ht_sample, hf_sample] = csinet_preprocess(H_freq, spec);
        HT(sample_idx, :) = ht_sample;
        HF_all(sample_idx, :, :) = hf_sample;
    end
end

function pos = sample_user_position(spec)
    half = spec.area_length_m / 2;
    switch spec.user_distribution
        case 'uniform'
            pos = (rand(1, 2) * 2 - 1) * half;
        case 'center'
            pos = max(min(randn(1, 2) * spec.area_length_m / 9, half), -half);
        case 'edge'
            theta = rand() * 2 * pi;
            radius = (0.65 + 0.35 * rand()) * half;
            pos = [radius * cos(theta), radius * sin(theta)];
        case 'ring'
            theta = rand() * 2 * pi;
            radius = min(max(0.55 * half + randn() * 0.08 * half, 0.30 * half), 0.85 * half);
            pos = [radius * cos(theta), radius * sin(theta)];
        case 'clustered'
            centers = [-0.45, -0.25; 0.35, 0.35; 0.20, -0.45] * half;
            center = centers(randi(size(centers, 1)), :);
            pos = max(min(center + randn(1, 2) * spec.area_length_m / 12, half), -half);
        otherwise
            error('Unsupported user distribution: %s', spec.user_distribution);
    end
end

function H_freq = cost2100_frequency_response(spec, user_pos)
    % Adapter for the official COST2100 MATLAB implementation.
    %
    % Expected output:
    %   H_freq: complex matrix [Nt, subcarriers]
    %
    % The public COST2100 model has several demos and parameter helpers. In
    % most checkouts, the entry point is cost2100(). If your local copy uses a
    % different parameter struct or returns a different field name, keep this
    % function as the only place to adapt.

    if exist('cost2100', 'file') ~= 2
        error('COST2100 MATLAB functions not found. Run addpath(genpath(<cost2100_root>/matlab)) first.');
    end

    scenario = 'LOS';
    if strcmp(spec.scenario, 'IndoorHall_5GHz')
        freq = [-10e6, 10e6] + 5.3e9;
        bs_pos = [0, 0, 0];
        ms_pos = [user_pos(1), user_pos(2), 0];
    elseif strcmp(spec.scenario, 'SemiUrban_300MHz')
        freq = [2.75e8, 2.95e8];
        bs_pos = [0, 0, 0];
        ms_pos = [user_pos(1), user_pos(2), 0];
    else
        error('Unsupported COST2100 scenario: %s', spec.scenario);
    end

    snap_rate = 1;
    snap_num = 1;
    bs_pos_spacing = [0, 0, 0];
    bs_pos_num = 1;
    ms_velo = [0, 0, 0];

    [~, ~, link, ~] = cost2100( ...
        spec.scenario, scenario, freq, snap_rate, snap_num, ...
        bs_pos, bs_pos_spacing, bs_pos_num, ms_pos, ms_velo);

    channel = link(1, 1).channel{1, 1};
    H_freq = channel_to_ula_frequency_response(channel, freq, spec.nt, spec.subcarriers);
end

function H_freq = channel_to_ula_frequency_response(channel, freq, nt, subcarriers)
    if isfield(channel, 'h') && isfield(channel, 'h_los')
        paths = [channel.h; channel.h_los];
    elseif isfield(channel, 'h')
        paths = channel.h;
    else
        error('COST2100 channel output does not contain field h.');
    end

    paths = paths(abs(paths(:, 6)) > 0, :);
    if isempty(paths)
        error('COST2100 returned no active paths.');
    end

    f_grid = linspace(freq(1), freq(2), subcarriers);
    antenna_idx = (0:nt - 1).';
    H_freq = complex(zeros(nt, subcarriers));

    for path_idx = 1:size(paths, 1)
        aod = paths(path_idx, 1);
        delay = paths(path_idx, 5);
        amp = paths(path_idx, 6);
        spatial = exp(-1j * pi * antenna_idx * cos(aod));
        frequency = exp(-1j * 2 * pi * f_grid * delay);
        H_freq = H_freq + spatial * (amp * frequency);
    end
end

function [ht_sample, hf_sample] = csinet_preprocess(H_freq, spec)
    % CsiNet reference preprocessing: angular-delay transform, keep first
    % 32 delay rows, split real/imag, and normalize to [0, 1].
    H_ad = fft(ifft(H_freq, [], 1), [], 2);
    H_ad = H_ad(:, 1:spec.keep_delay_rows).';

    scale = prctile(abs(H_ad(:)), 99.5);
    if scale < eps
        scale = 1;
    end
    H_ad = H_ad / scale;
    H_ad = complex(max(min(real(H_ad), 1), -1), max(min(imag(H_ad), 1), -1));

    real_part = single(real(H_ad) + 0.5);
    imag_part = single(imag(H_ad) + 0.5);
    real_part = max(min(real_part, 1), 0);
    imag_part = max(min(imag_part, 1), 0);
    ht_sample = reshape(cat(3, real_part, imag_part), 1, []);

    hf_sample = single(H_freq(:, 1:spec.keep_frequency_bins));
end
