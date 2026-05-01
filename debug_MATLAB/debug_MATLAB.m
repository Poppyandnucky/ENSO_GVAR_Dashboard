function debug_MATLAB()

%% STRUCTURAL BREAKS
%  Flag: flag.fig_pdf     (below, not necessary, takes time)
%        flag_plot_detail (in f_results(), only if needed overwrites Figure 99)
%        flag_write...

%% HEIRARCHICAL BAYESIAN APPROACH
%  Flags: 0 0 1 0 0 0 0 1 1 (for ENSO) 1 (clustering)
%  Clustering occurs here:
%    KF IN SVD SPACE
%    if flag.KF_in_SVD || flag.macro
%  Clustering variables:
%    Impact on CPI (h=0)
%    Impact on EX (h=0)
%    Peak GDP impact
%    Peak FX impact
%    Time-to-peak GDP
%    Cumulative GDP loss at 4 quarters
%    Cumulative CPI effect at 4 quarters
%  Regress commodity on ENSO in load_country_VAR_data() by setting flag_comm_regress=1
%    Result: beta ( ENSO -> COMMODITY_YoY ) = -0.286, p < 0.0001

%% ASSUMPTIONS
%  B non-zero ENSO effects:              ENSO -> CPI (up), EX (down)
%  A non-zero off-diagonal coefficients: GDP  -> EX
%                                        FX   -> CPI
%% CLIMATE EFFECTS
%  Egypt:     95% irrigation so precipitation has little effect except if it affects the Nile

%% TESTS
%  Load Python data
flag_load_python = 0;
if flag_load_python
    % terminate(pyenv)
    % pyenv(Version="/Users/ti/Sites/GitHub/TRP/.venv/bin/python3.11")
    npData = py.numpy.load('../data_MATLAB/data.npz');
    X      = double(npData.get('X')); % endogenous
    Y      = double(npData.get('Y')); % exogenous X_t+1 = A*X_t + B*Y_t + u_t
    p      = 1;
    res    = estimate_VARX(X,Y,p);
    [k,q]       = size(res.B);
    Beta0       = [ eye(k) ; zeros(k*(p-1),k) ; zeros(q,k) ];   % (n,k) used for KF
    Sigma_u     = res.Sigma_u; ZZ = res.ZZ;

    lambda      = 0.99;
    shrink      = 10;            % 1 is too small (slightly closer to Beta0)
    eps_scale   = 0.001;
    R_mat       = 1;

    P0                      = f_P0(ZZ,Sigma_u,p,shrink,eps_scale);
    [ A_KF,B_KF,beta_filt ] = f_KF(Y,X,Beta0,P0,Sigma_u,R_mat,lambda,p);
    
    fprintf('\nA from VARX\n')
    fprintf('%5.1f %5.1f %5.1f %5.1f %5.1f\n',res.A)
    fprintf('\nA from KF\n')
    fprintf('%5.1f %5.1f %5.1f %5.1f %5.1f\n',A_KF)
    fprintf('\nB from VARX\n')
    fprintf('%5.1f %5.1f %5.1f %5.1f\n',res.B)
    fprintf('\nB from KF\n')
    fprintf('%5.1f %5.1f %5.1f %5.1f\n',B_KF)

    keyboard, return
end

flag.simulate_XY = 0;
if flag.simulate_XY
    p        = 2; k = 2; q = 2; T = 3000;
    
    [ Y,X,A,B ] = generate_VARX(k,q,p,T);
    res         = estimate_VARX(Y,         X,p);
    res_no_X    = estimate_VARX(Y,zeros(T,0),p);

    Beta        = res.Beta; Sigma_u = res.Sigma_u; Z = res.Z; ZZ = res.ZZ;

    flag.use_A_sim = 0; % 1-use A_sim for debugging
    if flag.use_A_sim
        A_no_X  = A;    % gives better result than usual VARX of A and B together
    else
        A_no_X  = res_no_X.A;
    end

    y_hist      = Y(1:p,  :);    % last p values of y used for initial projection (or zeros)
%                                % use to project at time t=p+1
    y_hat       = forecast_varx(A_no_X,[],y_hist,[],T-1); % project to time (T-1)+1
    y_residual  = Y - y_hat;     % residual including X effects + u for estimating B
    res_no_Y    = estimate_VARX(y_residual, X, 0);   % use p=0 to estimate B only
    Beta_no_X   = [ A_no_X(:,:)' ; res_no_Y.B' ];    % append B based on residuals y_hat

    Beta0       = [ eye(k) ; zeros(k*(p-1),k) ; zeros(q,k) ];   % (n,k) used for KF
    
    lambda      = 0.99;
    shrink      = 10;            % 1 is too small (slightly closer to Beta0)
    eps_scale   = 0.001;
    R_mat       = 1;

    P0                      = f_P0(ZZ,Sigma_u,p,shrink,eps_scale);
    [ A_KF,B_KF,beta_filt ] = f_KF(X,Y,Beta0,P0,Sigma_u,R_mat,lambda,p);

    Beta_sim                    = [ A(:,:)'    ; B'    ];
    Beta_KF                     = [ A_KF(:,:)' ; B_KF' ];

    fprintf('\n Beta simulated:\n'), fprintf('%6.2f %6.2f\n',Beta_sim')
    fprintf('\n Beta KF:\n'       ), fprintf('%6.2f %6.2f\n',Beta_KF')
    fprintf('\n Beta VARX A,B:\n' ), fprintf('%6.2f %6.2f\n',Beta_no_X')
    fprintf('\n Beta VARX:\n'     ), fprintf('%6.2f %6.2f\n',Beta')

    keyboard, return
end

%% ENSO IRF SETUP
flag_setup_irf = 1;
if flag_setup_irf
    f_setup_irf();
end

%% FLAGS
flag.SVD_A_only    = 0; % 0 use SVD for A and retain all coefficients in B
flag.simulate_data = 0; % 0
flag.macro         = 1; % 0 use macro assumptions (not too sensitive)
flag.KF_in_SVD     = 0; % 1 perform KF in SVD space (uses nSVD1)
flag.KF_projected  = 0; % 0 project KF in SVD coordinates
flag.SVD           = 0; % 1
flag.macro_no_X    = 0 & flag.macro; %  1   use macro assumptions in the no_X analysis
j_ex             = 1;              %  2
shock_size       = 1;              % -1 for moisture, +1 for enso => food inflation
% j_ex               = [1      2];     %  1 2 for enso, commodities
% shock_size         = [1 -0.286];     %    dual shock
flag.clustering    = 1;
flag.fig_pdf       = 0;

%% SIMULATE DATA
T_sim              = 100; % 100, 300, 1000, 3000
flag.create_csv    = 0;
if  flag.simulate_data
    % Only a subset of simulated X is selected by f_results()
    Y_names_sim     = [ "GDP_YoY","CPI_YoY","FX_YoY","EX_YoY" ];
    X_names_sim     = [ "ENSO","PRITHVI_MOISTURE_STD","PRITHVI_HEAT_STD" ];
    csv_file        = sprintf('../data/gvar_panel_streamlit_sim_%i.csv',T_sim);

    A           =  0.7*eye(4);
    A(1,2)      = -0.1;
    A(2,3)      =  0.1;
    A(3,4)      =  0.15;
    B           = zeros(4,numel(X_names_sim));
    % Coordinates correspond to X_names_sim
    B(1,2)      =  0.5;              % X ~ N(0,I)
    u_std       =  0.1;              % u ~ N(0,I*u_std)

    if flag.create_csv
        countries   = [ "BRA","CHL","COL","IDN","IND","KEN","MEX","PER","PHL","THA","ZAF" ];
        n_countries = numel(countries);
        X           = []; Y = []; c = []; d = [];
        nX_sim      = numel(X_names_sim);
        nX_sim_common = 1;               % simulated ENSO and other global indices
        X_common    = randn(T_sim,nX_sim_common);

        for i=1:n_countries
            Xi = [ X_common randn(T_sim,nX_sim - nX_sim_common) ];
            Yi = zeros(T_sim,numel(Y_names_sim));
            ui = randn(T_sim,4)*u_std;
            for t=2:T_sim
                % y*D   = y*D*A   + x*B   + u (D = std(Y), std(X) = 1)
                % y     = y*D*A/D + x*B/D + u/D
                % A_mdl = D*A/D
                % B_mdl =   B/D
                Yi(t,:) = Yi(t-1,:)*A' + Xi(t-1,:)*B' + ui(t-1,:);
            end
            ci = repmat(countries(i),T_sim,1);
            di = datetime(2025,10,1) + calquarters(-T_sim+1:0)';
            c  = [ c ; ci ];
            Y  = [ Y ; Yi ];
            X  = [ X ; Xi ];
            d  = [ d ; di ];
        end

        d.Format = 'M/d/yyyy';
        Tbl = table(c, d, year(d), Y, X, 'VariableNames', ...
            {'country','quarter','year','Y','X'});
        Tbl = splitvars(Tbl, 'X', 'NewVariableNames', X_names_sim);
        Tbl = splitvars(Tbl, 'Y', 'NewVariableNames', Y_names_sim);
        writetable(Tbl, csv_file);
        % keyboard
    end
end
if ~flag.simulate_data
    csv_file = '../data/gvar_panel_streamlit.csv';
end

%% INITIALIZE PARAMETERS
%  load_country_VAR_data() has initial data selection, subsequent section is below (idx_XY)
p        = 1;
% load var_input_data1.mat % Python file
% p  = double(p);

%  IRF FOR ENSO (see f_irf_X())
%  ENSO_t = phi*  ENSO_t-1 + noise
%  COMM_t = gamma*COMM_t-1 + delta_0*ENSO_t + delta_1*ENSO_t-1 + noise

% idx_X   = [ 5 15:16 ]; % ENSO + ENSO1,2
% idx_X   = [ 5 11    ]; % ENSO + moisture extent
% idx_X   = [   10    ]; %        moisture std
% idx_X   = [   17    ]; %        moisture std1 (6=US_GDP, 8=commodity)
% idx_X   = [   15    ]; % ENSO1
% idx_X   = [1:5 11:12]; % star
% idx_X   = [  5 11:12]; % ENSO + climate
% idx_X   = [  5      ]; % ENSO (unstable countries)
idx_X   = [ 15  8 6   ]; % ENSO1, commodities (ENSO1 better than ENSO), US_GDP
idx_Y   = [ 1 2 4 3   ]; % GDP CPI EX FX *** CPI SEEMS MORE REASONABLE THAN CPIF ***
idx_XY  = {idx_X,idx_Y};
jBkeep  = [   2 3     ]; % 1:4 indices of B (ENSO        -> A) to keep
jBkeep  = [ jBkeep 6:8]; % 5:8              (commodities -> A)
jBkeep  = [ jBkeep 9:12];% US GDP

% shrink    = 1e-3;   % example, too small
% shrink    = 1;      % no shrinkage
% shrink    = 0.03;   % appropriate given that P0 ranges from 0.01 - 0.06
% eps_scale = 1e-8;

% P0*shrink uncertainty in A,B
% lambda forgetting factor
% R variance of innovation of y
lambda      =    1; shrink = 1; eps_scale = 0.001; R_mat = 1; % is more stable than VAR A,B
% shrink    = 0.04/4;
% lambda    = 0.98;
R_mat       = 10;
% lambda    = 0.98; shrink = 1; eps_scale = 0.001; R_mat = 1; % is unstable for KF_in_SVD

%% RESULTS
% lambda = 1.00; shrink = 1.00; eps_scale = 0.001; R_mat = 10; % reasonable
%    beta_mean_select = [ 3 4 7 8 11:17 ]                      % not much AUS effect on CPI
% lambda = 0.98                                                %          AUS -> high CPI
% R_mat  =    1                                                % AUS,BRA high CPI
% R_mat  =  100                            % too much constraint, PAK unstable

hor       = 12;             % horizon

T_covid_end     = 20*0;     % T_covid = T - T_covid_end; X = X(1:T_covid-1,:);

flag.plot_irf   = 1;
fig_no_irf      = 101;      % figure for plotting data
flag.plot       = 0;

%  Get countries, variables names, parameters (initialize using input countries = [])
[ countries,Y_names,X_names,k,q,n,nb ] = ...
    f_results(csv_file,[],p,idx_XY,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,flag.plot); 

if ismember(5,idx_XY{2})    % only a subset of countries have CPIF
%   countries   = countries(1+[0 3 6 7 10 11]); % countries(5) IND has only 21 quarters of CPIF
    countries   = countries([2 5:8 12:15 17 20:22]-1);
else
%   countries   = countries(1:10);
    % NEED TO FIX FX IN SPAIN [DONE]
    % countries   = countries(~strcmp(countries,'ESP'));
    % countries   = countries(1);
    % countries   = countries(setdiff(1:numel(countries),4)); % take out 4th country [OLD]
end
n_countries = numel(countries);

%  Make sure flags are consistent
if flag.KF_in_SVD || flag.KF_projected, flag.SVD = 1;                              end
if flag.macro,                          flag.SVD = 0;                              end
if flag.SVD==0;                         flag.KF_in_SVD = 0; flag.KF_projected = 0; end

%% MULTIPLE COUNTRY ANALYIS: KF and VARX
if p==0                             % no Y, only X
    Beta0  = zeros(q,k);
else
    Beta0  = [ eye(k) ; zeros(k*(p-1),k) ; zeros(q,k) ];   % Y and X (n,k)
end
Beta0      = Beta0 * 0.25;          % scale identity
A_KFs      = nan(k,k,max(1,p),  0); % allow zeros if p=0 [REPLACE n_countries WITH 0]
A_KFs_no_X = A_KFs;
B_KFs      = nan(k,q,           0);
B_KFs_no_X = B_KFs;
if q>0
    y_diff_KFs = nan(hor+1,k,0);
else
    y_diff_KFs = nan(0,    0,0);
end
Betas      = nan(n,k,    0);
Betas_no_X = zeros(n,k,  0);
beta_filts = cell(1,     0);        % each cell is nb,T
beta_filts_no_X = beta_filts;
ys         = cell(1,     0);
xs         = cell(1,     0);
DYs        = cell(1,     0);

%  Get A,B,beta_filt from KF, Beta from VARX, all models based on x,y (not X,Y)
%  fig = figure(150); fig.Name = 'residuals'; tiledlayout(3,n_countries)

i   = 1;                                      % initialize the index
i_not_enough_data = [];
n_countries0      = n_countries;
countries0        = countries;
for I=1:n_countries0
    i_good =      0;                          % set false unless the data is good
    try
        [~,~,~,~,~,~,~,Ts(i),A_KFs(:,:,:,i),B_KFs(:,:,i),y_diff_KFs(:,:,i),...
            Betas(:,:,i),beta_filts{i},ys{i},xs{i},DYs{i}] = f_results(csv_file,countries{i},...
            p,idx_XY,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,0,0,[],Beta0,...
            [],[],[],[],I*flag.fig_pdf);
        if ~isnan(Betas(1,1,i)), i_good = 1; end
    catch ME
        if ~strcmp(ME.identifier,'MATLAB:unassignedOutputs'), keyboard, end
    end
    if i_good==0
        i_not_enough_data = [i_not_enough_data i];
        countries   = countries(setdiff(1:n_countries,i));
        n_countries = n_countries - 1;
        i_good      = 0;
        continue
    end
    if ~flag.macro_no_X
        %  A without B (no X)
        idx_0Y = {[],idx_XY{2}};
        [~,~,~,~,~,~,~,~,    A_KFs_no_X(:,:,:,i),~,~,...
            Betas_no_X(1:k*p,:,i),beta_filts_no_X{i},~,~,~,~,dates] = ...
            f_results(csv_file,countries{i},...
            p,idx_0Y,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,0,0,[],Beta0(1:k*p,:));

        yi         = ys{i};
        yi_hat_KF  = forecast_varx(A_KFs_no_X(:,:,:,i),   [], yi(1:p,:), [], Ts(i)-1);
        yi_hat_VAR = forecast_varx(Betas_no_X(1:k*p,:,i), [], yi(1:p,:), [], Ts(i)-1);
        yi_residual_KF  = yi - yi_hat_KF;                      % used to estimate B
        yi_residual_VAR = yi - yi_hat_VAR;

        % estimate B without A using the residuals from y_t - A*y_{t-1}
        res_i_KF   = estimate_VARX(yi_residual_KF,  xs{i}, 0); % p=0, estimate B only
        res_i_VAR  = estimate_VARX(yi_residual_VAR, xs{i}, 0);
        Betas_no_X(k*p+1:n,:,i) = res_i_VAR.B';
        B_KFs_no_X(:,:,i)       = res_i_KF.B;
        d                       = dates.quarter(end-Ts(i)+1:end)+calyears(2000);
%       nexttile(i), plot(d,yi_residual_VAR), title(countries{i})
%       xtickformat('yyyy');
%       nexttile(n_countries  +i), plot(d,xs{i}(:,1)), xtickformat('yyyy')
%       nexttile(n_countries*2+i), plot(d,xs{i}(:,2)), xtickformat('yyyy') % ENSO
    end
    if i_good, i = i+1; end          % increment index and continue
end

if ~isempty(i_not_enough_data)
    fprintf('Countries with not enough data:\n'), fprintf('%s, ',countries0(i_not_enough_data))
    fprintf('\n')
end

%% Betas FROM ALL COUNTRIES
T_min      = min(Ts);                % start w/83 countries, reduce to 71 (not enough data)
Betas_mean = mean(Betas,3);          % average betas from standard VAR

%  Plot Betas from KF, VARX
clims      = [-0.5 0.8];
if flag.simulate_data && q > 0
    j_ex_sim = find(X_names_sim==X_names);
    Beta_sim = [A' ; B(:,j_ex_sim)'];
else
    Beta_sim = [];
end

%% STACKED SVD OF VARX OVER STAGGERED TIME SPANS
%  Closest to non-SVD: flag.stack_VAR = 0, flag.stack_beta_file = 1;
flag.stack_VAR       = 0;           % 0-stack only one beta for each of the 11 countries
%                                   % 1-use VARX w/staggered times (t_duration=0 => all time)
flag.stack_beta_filt = 1;           % 0-use Betas from VARX
%                                   % 1-use beta_filt from KF, 

if  flag.stack_VAR
    %  Use VARX at staggered times
    t_duration    = 4*5;            % number of quarters for VAR for generating SVD
    beta_stacked  = nan(0,nb);
    for i=1:n_countries
        t_end = Ts(i);              % 1st time interval: last t_duration t
        while t_end>=t_duration     % make sure we have enough data
            if flag.stack_beta_filt
                beta_i = beta_filts{i}(:,t_end)';
            else
                if t_duration==0
                    t_start = 1;                       % take all of the data
                else
                    t_start = t_end - t_duration + 1;  % start of time interval
                end
                yi       = ys{i}(t_start:t_end,:);     % collect data
                xi       = xs{i}(t_start:t_end,:);
                res      = estimate_VARX(yi, xi, p);
                % Beta from VARX is [ A' ; B' ]
                % Z_t-1 is a row vector [ Y_t-1 Y_t-2 ... Y_t-p X_t-1 ]
                % Z_t-1 * Beta ~ Y_t(1,k)
                % A_i    = res.A;                      % (k,k*p)
                % B_i    = res.B;
                % AB_i   = [ A_i(:) ; B_i(:) ];
                beta_i   = res.Beta(:)';               % same order as Zt from build_Zt()
            end
            beta_stacked = [ beta_stacked ; beta_i ];
            if t_duration==0
                t_end    = -1;                     % end iteration
            else
                t_end    = t_end - t_duration;     % go backwards in time
            end
        end
    end
end
if ~flag.stack_VAR
    %  LOOKS GOOD
    %  Use final A,B from KF
    %  Stack these rows from final KF time: [ A(:,1) ; ... ; A(:,k) ; B(:,1) ; ... ; B(:,q) ]'
    beta_stacked = nan(n_countries,nb); % n = p*k + q, nb = n*k
    for i=1:n_countries
        if flag.stack_beta_filt
            beta_stacked(i,:) = beta_filts{i}(:,round(Ts(i)*0.8));
%           Similar
%           beta_stacked(i,:) = reshape([ A_KFs(:,:,1,i)' ; B_KFs(:,:,i)' ],1,nb);
        else
            beta_stacked(i,:) = reshape(Betas(:,:,i),1,nb);
        end
    end
end

%% SUBTRACT THE MEAN (beta_stacked_mean) AND GET SVD OF VARIATION FROM THE MEAN

%% USE SELECT COUNTRIES
for i=1:n_countries, fprintf('%2i %s\n',i,countries(i)), end

beta_mean_select= 1:n_countries;
% beta_mean_select  = [2 4 8 9 15 16];
% beta_mean_select  = [3 4 7 8 11:17];
beta_stacked_mean = mean(beta_stacked(beta_mean_select,:));
beta_stacked      = beta_stacked - beta_stacked_mean;

if flag.SVD
    %% OPTION FOR SVD A ONLY
    if flag.SVD_A_only
        % remove B components before SVD and insert I(q*k) later (1st q*k SVD vectors)
        [ ~,S_stacked,V_stacked0 ] = svd(beta_stacked(:,1:(nb-q*k))); % AB = U*S*V'
    else
        [ ~,S_stacked,V_stacked  ] = svd(beta_stacked); % AB = U*S*V'
    end

    n_SVD0    = 5+15;                        % number of SVD matrices to retain initially
    n_SVD0    = min(n_SVD0,min(size(S_stacked)));
    n_V       = size(S_stacked,2);           % number of rows in V
    s_stacked = zeros(n_SVD0,1);
    s_stacked = diag(S_stacked(1:n_SVD0,:)); % used in figure(132)

    %% INSERT I(q*k)
    if flag.SVD_A_only
        n_SVD0    = n_SVD0 + q*k;                     %
        s_stacked = [ ones(q*k,1)    ; s_stacked      ];
        V_stacked = zeros(nb);
        idx_A     = find(mod(1:nb,n));
        col_V     = 0;
        col_V0    = 0;
        for j=1:k
            for l=1:q
                col_V = col_V + 1;
                V_stacked(j*n+l-1,col_V) = 1;
            end
        end
        for j=1:k
            for l=1:k
                col_V  = col_V + 1;
                col_V0 = col_V0 + 1;
                V_stacked(idx_A,col_V) = V_stacked0(:,col_V0);
            end            
        end
        % V_stacked = [ zeros(n_V,q*k)   V_stacked0
        %               eye(      q*k)   zeros(q*k,n_V) ];      % (NG)
    end

    beta_SVD  = V_stacked(:,1:n_SVD0);       % (nb,n_SVD)

    Beta_SVD               = reshape(beta_SVD,         n,k,n_SVD0);  % unstack into (p*k+q,k) matrix
    Beta_stacked_mean_plot = reshape(beta_stacked_mean,n,k);

    %  Plot AB from SVD
    fig = figure(132); fig.Name = 'A_stacked_SVD'; tiledlayout('flow'), clf
    nexttile
    imagesc(Beta_stacked_mean_plot), colorbar, title('Mean')
    for j=1:n_SVD0
        nexttile
        imagesc(Beta_SVD(:,:,j)*s_stacked(j)), colorbar, title(sprintf('SVD %i',j))
    end
    nexttile, plot(s_stacked,'bo-'), title('Singular values') 
end

%% PROJECTIONS OF KF IN ORIGINAL SPACE (beta_filt) ONTO SVD SPACE
if flag.KF_projected
    %  Take the last T_min time values for each beta_filt
    beta_filt_T_min = nan(nb,T_min,n_countries);
    for i=1:n_countries
        beta_filt_T_min(:,:,i) = beta_filts{i}(:,end-T_min+1:end);
    end

    %  Need to make sure there are no non-zero mean components before projection
    flag.mean_original_SVD = 1;                        % 0 is better, zero-mean worst
    if flag.mean_original_SVD
        beta_filt_T_min_mean = beta_stacked_mean';                 % original SVD mean
    else
        beta_filt_T_min_mean = mean(beta_filt_T_min(:,end,:),3);   % avg over countries, last time point
    end
    beta_filt_T_min_del = beta_filt_T_min - beta_filt_T_min_mean;  % variation from the mean
    beta_filt_T_min_SVD = nan(n_SVD0,T_min,n_countries);           % projected in SVD space
    for i=1:n_countries
        %   beta_filt_T_min_SVD(:,:,i) = diag(1./s_stacked(1:n_SVD0)) * ...
        beta_filt_T_min_SVD(:,:,i) = ...
            V_stacked(:,1:n_SVD0)' * beta_filt_T_min_del(:,:,i);   % (n_SVD0,T_min,n_countries)
    end
    %  Get projection using
    %    V_stacked(:,1:n_SVD0) * beta_filt_T_min_SVD(:,:,i) + beta_filt_T_min_mean

    %  Plot time trajectories of beta_filt in SVD space
    fig = figure(133); fig.Name = 'beta_filt_SVD'; tiledlayout('flow'), clf
    for i=1:n_countries
        nexttile
        plot(1:T_min,beta_filt_T_min_SVD(:,:,i)), title(countries{i})
    end

    %  Betas at final time projected back in original space
    Beta_filtP = reshape(V_stacked(:,1:n_SVD0) * squeeze(beta_filt_T_min_SVD(:,end,:)), ...
        n,k,n_countries) + reshape(beta_filt_T_min_mean,n,k);
    plot_Beta(141,'Betas: KF projected',Beta_filtP,[],[],DYs,clims,countries,Beta_sim)
end

%% KF IN SVD SPACE
if flag.KF_in_SVD || flag.macro
    if flag.KF_in_SVD
        %  When using all SVD modes, results are slightly different because of beta_stacked_mean
        n_SVD1      = 20;                               % number of SVD matrices to retain
        n_SVD1      = min(n_SVD1,n_SVD0);
    end
    if flag.macro
        n_SVD1      = k+2+numel(jBkeep)+2;              % diag, 2 in A, * in B, 2 more in A
        sz          = [n,k];                            % size of Beta
        V_stackedA  = zeros(nb,n_SVD1);                 % corresponds to [ A' ; B' ](:)
        V_stackedB  = zeros(nb,n_SVD1);
        for j=1:k
            V_stackedA(sub2ind(sz,j,j), j) = 1;         % 1, n+2, 2*n+3, ...
        end
        %  rows of Beta: GDP, CPI, FX, EX, ENSO (input)
        %  cols of Beta: GDP, CPI, FX, EX       (output)
%       V_stackedA(sub2ind(sz,1,3), k + 1) = 1;         % GDP (row 1) -> EX (col 3)
%       V_stackedA(sub2ind(sz,4,2), k + 2) = 1;         % FX  (row 4) -> CPI(col 2) 
        V_stackedA(sub2ind(sz,3,1), k + 1) = 1;         % EX  (row 3) -> GDP(col 1)
        V_stackedA(sub2ind(sz,2,4), k + 2) = 1;         % CPI (row 2) -> FX (col 4) 
        j_Vcol       = k + 2;                           % number of columns in V_stacked
        for j=1:numel(jBkeep)
            j_Vcol   = j_Vcol + 1;
            [ j_row,j_col ] = ind2sub([k,q],jBkeep(j)); % ENSO(row n) -> endogenous        
            V_stackedB(sub2ind(sz,j_col+k,j_row), j_Vcol) = 1;
        end

        V_stackedA(sub2ind(sz,2,1), j_Vcol+1) = 1;      % CPI (row 2) -> GDP(col 1) *
        V_stackedA(sub2ind(sz,3,4), j_Vcol+2) = 1;      % EX  (row 3) -> FX (col 4) *
        beta_stacked_mean         = zeros(1,nb);
        % [ A_debug,B_debug ]     = extract_AB(V_stackedB(:,k+3),p,k,q)
        % [ A_debug,B_debug ]     = extract_AB(V_stackedB(:,k+4),p,k,q)
        V_stacked = V_stackedA + V_stackedB;
    end

    A_KFVs      = nan(k,k,max(1,p),n_countries);
    B_KFVs      = nan(k,q,         n_countries);
    BetaVs      = nan(n,k,         n_countries);
    beta_filtVs = cell(1,     0);                         % each cell is nb,T
    a_filtVs    = cell(1,          n_countries);
    if q>0
        y_diff_KFVs = nan(hor+1,k, n_countries);
    else
        y_diff_KFVs = nan(0,0,     n_countries);
    end

    A_KFs_no_Xm = nan(k,k,max(1,p),n_countries);          % allow zeros if p=0
    B_KFs_no_Xm = nan(k,q,         n_countries);
    Betas_no_Xm = zeros(n,k,       n_countries);
    beta_filts_no_Xm = cell(1,     n_countries);          % each cell is nb,T

    %  If n_SVD = nb, mean can be replaced by zeros or random vector
    %  beta_stacked_mean is added back to beta_filts, A_KFVs, B_KFVs
    flag_mean_select = 1;
    if flag_mean_select, n_iter = 2; end
    beta_stacked_means = repmat(beta_stacked_mean,n_countries,1);
    beta_filtVs_end    = nan(n_countries,nb);
    beta_sigmas        = nan(n_countries,nb);

    for i_iter=1:n_iter
        for i=1:n_countries
            % a_filtVs{i}        are SVD coordinates of beta
            % y_diff_FKVs(:,:,i) are IRF using KFV (baseline - shock responses)
            [~,~,~,~,~,~,~,~,A_KFVs(:,:,:,i),B_KFVs(:,:,i),y_diff_KFVs(:,:,i),...
                BetaVs(:,:,i),beta_filtVs{i},~,~,~,~,~,~,~,~,a_filtVs{i}] = ...
                f_results(csv_file,countries{i},...
                p,idx_XY,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,0,0,[],Beta0,...
                V_stacked,n_SVD1,beta_stacked_means(i,:),beta_sigmas(i,:));
            beta_filtVs_end(i,:) = beta_filtVs{i}(:,end)';
            if flag.macro_no_X
                %  A without B (no X), V_stackedA has zeros corresponding to B
                [~,~,~,~,~,~,~,~,    A_KFs_no_Xm(:,:,:,i),~,~,...
                    Betas_no_Xm(:,:,i),beta_filts_no_Xm{i}]      = f_results(csv_file,countries{i},...
                    p,idx_XY,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,0,0,[],Beta0,...
                    V_stackedA,n_SVD1,beta_stacked_mean);

                yi         = ys{i};
                yi_hat_KFm  = forecast_varx(A_KFs_no_Xm(:,:,:,i),   [], yi(1:p,:), [], Ts(i)-1);
                yi_hat_VARm = forecast_varx(Betas_no_Xm(1:k*p,:,i), [], yi(1:p,:), [], Ts(i)-1);
                yi_residual_KFm  = yi - yi_hat_KFm;                      % used to estimate B
                yi_residual_VARm = yi - yi_hat_VARm;
                res_i_KFm   = estimate_VARX(yi_residual_KFm(:, jBkeep),xs{i},0); % p=0, est B only
                res_i_VARm  = estimate_VARX(yi_residual_VARm(:,jBkeep),xs{i},0);
                B = zeros(k,q); B(jBkeep,:) = res_i_KFm.B;  res_i_KFm.B  = B;
                B = zeros(k,q); B(jBkeep,:) = res_i_VARm.B; res_i_VARm.B = B;
                Betas_no_Xm(k*p+1:n,:,i) = res_i_VARm.B';
                B_KFs_no_Xm(:,:,i)       = res_i_KFm.B;
            end
        end

        % 0.10 => 27 out of 77 countries
        % 0.25    57
        lim_keep            = 0.25;                                  % decay upper limit at hor
        [ idx_keep,n_keep ] = analysis_A(A_KFVs,hor,lim_keep);

        %% SECOND ITERATION, USE DIFFERENT beta_stacked_mean USING beta_mean_select COUNTRIES
        if n_iter>1 && i_iter==1
            if flag.clustering
                n_SVD_cl  = min(3,k);                       % number of SVD components
                n_cl      = 3;                              % number of clusters

                y_diff_cl = cell(1,n_countries);
                y_cl0     = nan( k,n_countries);

                % 
                for i=1:n_countries
                    y_diff_cl{i}  = f_irf_X(j_ex,shock_size,hor,A_KFVs(:,:,1,i),[],B_KFVs(:,:,i));
                    y_cl0(2:3,i)  =         y_diff_cl{i}(2,  2:3);   % CPI,EX
                    [ ~,t_max ]   = max(abs(y_diff_cl{i}(:,    1)));
                    y_cl0(1,  i)  =         y_diff_cl{i}(t_max,1);   % peak GDP
                    [ ~,t_max ]   = max(abs(y_diff_cl{i}(:,    4)));
                    y_cl0(4,  i)  =         y_diff_cl{i}(t_max,4);   % peak FX
                end
                y_cl  = (y_cl0 - mean(y_cl0(:,idx_keep),2)) ./ std(y_cl0(:,idx_keep),[],2);
                flag_sign = 1;
                if flag_sign
                    for i=1:n_countries
                        y_cl(2:3,i) = sign(y_diff_cl{i}(2,2:3));     % sign of CPI,EX
                    end
                end
                % U(n_countries)*S(n_countries,k)*V(k,k)' = y_cl'(n_countries,k)
                [ U_cl,S_cl,V_cl ] = svd(y_cl(:,idx_keep)');         % SVD (more stable)
                U_cl  = U_cl(:,1:n_SVD_cl);                          % n_SVD_cl components
                U1_cl =  [ U_cl
                    repmat(U_cl(Ts(idx_keep)>=40,:),4,1)
                    repmat(U_cl(Ts(idx_keep)>=80,:),4,1) ];          % duplicate better data

                flag_hierarchical = 0;
                if flag_hierarchical
                    D1_cl = pdist(  U_cl ,"euclidean");
                    Z1_cl = linkage(D1_cl,"complete");               % ward yields orphans
                    C1_cl = cluster(Z1_cl,"maxclust",3);
                    
                    fig   = figure(141); fig.Name = 'dendrogram'; dendrogram(Z1_cl);
                end
                if ~flag_hierarchical
                    % kmedoids handles noisy rows much better than kmeans
                    [C1_cl, M_cl] = kmeans(U1_cl, n_cl, 'Distance', 'cityblock');
                end
                C_cl            = ones(n_countries,1) * (n_cl+1);    % default: overall mean
                C_cl(idx_keep)  = C1_cl(1:n_keep);                   % don't include repeats

                for k_cl=1:n_cl+1                                    % for each cluster
                    if k_cl<n_cl
                        idx = C_cl==k_cl;                            % index of countries
                    else
                        idx = idx_keep;                              % all good countries
                    end
                    % save the mean and std for that cluster
                    beta_stacked_means(idx,:) = repmat( ...
                        mean(beta_filtVs_end(idx,:),   1), sum(idx), 1);
                    beta_sigmas(       idx,:) = repmat( ...
                        std( beta_filtVs_end(idx,:),[],1), sum(idx), 1);
                end
            end
            if ~flag.clustering
                fprintf('\nB from KF at last time step:\n')
                for i=1:n_countries
                    if ismember(i,beta_mean_select), s = '*'; else, s = ' '; end
                    fprintf('%2i%s %6.3f %6.3f %6.3f %6.3f\n',...
                        i,s,beta_filtVs_end(i,n-q+1:n:end))
                    for j=2:q
                        fprintf('    %6.3f %6.3f %6.3f %6.3f\n',...
                            beta_filtVs_end(i,n-q+j:n:end))
                    end
                end

                beta_stacked_mean  = mean(beta_filtVs_end(beta_mean_select,:));
                beta_stacked_means = repmat(beta_stacked_mean,n_countries,1);
            end
        end
    end

    %  Plot of SVD components (convert back from beta_filtVs)
    fig = figure(134); fig.Name = 'Betas over time, SVD'; tiledlayout('flow'), clf
    for i=1:n_countries
        nexttile
        plot(1:Ts(i),V_stacked(:,1:n_SVD1)'*beta_filtVs{i})
        title(countries{i})
    end

    %  Plot of Beta from KF in SVD space
    plot_Beta(135,'Betas: KF in SVD space',[],A_KFVs,B_KFVs,[],clims,countries,Beta_sim)

    SVD_coeffs = nan(n_SVD1,n_countries);
    for i=1:n_countries, SVD_coeffs(:,i) = a_filtVs{i}(:,end); end

    %  Bar plots
    fig = figure(136); fig.Name = 'Coeff final, SVD'; tiledlayout('flow'), clf
    nexttile, bar(SVD_coeffs','stacked')
    title('SVD coefficients'), xticks(1:n_countries), xticklabels(countries)

    if flag.KF_projected
        nexttile, bar(squeeze(beta_filt_T_min_SVD(1:n_SVD1,end,:))','stacked')
        title('projected SVD coefficients'), xticks(1:n_countries), xticklabels(countries)
    end
end

%% PLOT BETAS
plot_Beta(130,'Betas: KF',         [], A_KFs,     B_KFs,     DYs, clims,countries,Beta_sim)
plot_Beta(140,'Betas: VARX',       Betas,         [],[],     DYs, clims,countries,Beta_sim)
if flag.macro_no_X
    plot_Beta(142,'Betas: KF A, VAR B',[], A_KFs_no_Xm,B_KFs_no_Xm,DYs, clims,countries,Beta_sim)
    plot_Beta(143,'Betas: VAR A,B',    Betas_no_Xm,    [],[],      DYs, clims,countries,Beta_sim)
else
    plot_Beta(142,'Betas: KF A, VAR B',[], A_KFs_no_X, B_KFs_no_X, DYs, clims,countries,Beta_sim)
    plot_Beta(143,'Betas: VAR A,B',    Betas_no_X,     [],[],      DYs, clims,countries,Beta_sim)
end

%% PLOT ENSO IRFS FROM DIFFERENT MODELS
if q==0, keyboard, return, end

hor_plot = 1:7;
fig      = figure(137); fig.Name = 'IRFs with KF using SVD'; clf

n_col_plot = n_countries;
if flag.simulate_data, n_col_plot = n_col_plot + 1; end
if n_countries>20, flag_plot_all = 0; else, flag_plot_all = 1; end

if flag_plot_all
    tiles = tiledlayout(4,n_col_plot,'TileSpacing','tight','Padding','tight');
else
    tiledlayout('flow','TileSpacing','tight','Padding','tight');
end

y_diffs = nan(n_countries,k);

for i=1:n_countries
    % KF in subspace with stacked mean from KF or VARX
    % KF and VARX results give very different results

    % BASELINE KF: final A,B: A_KFs (mean value only used in SVD)
    % if flag_plot_all=0, plots may still be generated in the second block of code
    if flag_plot_all
        y_diff = f_irf_X(j_ex,shock_size,hor,A_KFs( :,:,1,i),[],B_KFs( :,:,i));
        nexttile(tilenum(tiles,1,i))
        plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
        if i==1, ylabel(sprintf('Baseline KF, nb = %i',nb)), end

        % PROJECTED KF: beta_filt onto SVD and back to original space (including the mean)
        if flag.KF_projected
            %     beta_filt_proj = V_stacked(:,1:n_SVD1)*beta_filt_T_min_SVD(1:n_SVD1,end,i) + beta_filt_T_min_mean;
            %     [ A,B ] = extract_AB(beta_filt_proj, p, k, q);
            %     y_diff  = f_irf_X(j_ex,shock_size,hor,A,[],B);
            %     nexttile(tilenum(tiles,2,i))
            %     plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
            %     if i==1, ylabel(sprintf('Projected KF, n_{SVD1} = %i',n_SVD1)), end
        end

        %% IRF BASED ON BETAS FROM VARX AND KF USING 
        %%   ENDOGENOUS MODEL TO GET A AND SUBSEQUENTLY USING THE RESIDUAL TO GET B
        %  A_no_X, B_no_X (from VARX without X)
        if flag.macro_no_X
            [ A_no_X_i,B_no_X_i ] = extract_AB(Betas_no_Xm(:,:,i), p, k, q);
        else
            [ A_no_X_i,B_no_X_i ] = extract_AB(Betas_no_X( :,:,i), p, k, q);
        end
        y_diff = f_irf_X(j_ex,shock_size,hor,A_no_X_i,[],B_no_X_i);
        nexttile(tilenum(tiles,2,i))
        plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
        if i==1, ylabel(sprintf('VAR A,B',nb)), end

        %  A_KFs_no_X, B_KFs_no_X (from KF without X)
        if flag.macro_no_X
            A_no_X_i = A_KFs_no_Xm(:,:,:,i); B_no_X_i = B_KFs_no_Xm(:,:,i);
        else
            A_no_X_i = A_KFs_no_X( :,:,:,i); B_no_X_i = B_KFs_no_X( :,:,i);
        end
        y_diff = f_irf_X(j_ex,shock_size,hor,A_no_X_i,[],B_no_X_i);
        nexttile(tilenum(tiles,3,i))
        plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
        if i==1, ylabel(sprintf('KF A, VAR B',nb)), end
    end

    %% KF IN SVD SPACE: A_KFVs based on n_SVD1
    %  A_KFVs, B_KFVs already transformed back to the original space in f_KF()
    if flag.KF_in_SVD || flag.macro
    % beta_stacked_mean is already added back to beta_filts, A_KFVs, B_KFVs
    %   in f_results > f_KF (when the SVD matrices are passed)
        y_diff = f_irf_X(j_ex,shock_size,hor,A_KFs( :,:,1,i),[],B_KFs( :,:,i),[],[],[],0);
%       y_diff = f_irf_X(j_ex,shock_size,hor,A_KFVs(:,:,1,i),[],B_KFVs(:,:,i));
        if flag_plot_all
            nexttile(tilenum(tiles,4,i))
        else
            nexttile
        end
        plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
        if i==1, ylabel(sprintf('KF in SVD space, n_{SVD1} = %i',n_SVD1)), end
        y_diffs(i,:) = y_diff(2,:);
    end
end

for i=1:n_countries
    fprintf('%.3f,%.3f,%.3f,%.3f,%s\n',y_diffs(i,:),countries{i})
end

if flag.simulate_data
    nexttile(tilenum(tiles,1,n_col_plot))
    IRF_X = f_irf_X(j_ex,shock_size,hor,A,[],B);
    plot(hor_plot,IRF_X(hor_plot,:),'LineWidth',1.5), title('Simulation')    
end

keyboard, return

flag.plot_map = 0; % NG
if flag.plot_map
    % 1. Load the comprehensive Natural Earth geojson file
    world = readgeotable("ne_110m_admin_0_countries.geojson");

    % 2. Your custom data table (ISO3 format)
    myData = table(["USA"; "FRA"; "CHN"; "BRA"; "ZAF"], [10; 20; 30; 40; 50], ...
        'VariableNames', ["ISO3", "Metric"]);

    % 3. Use your Dictionary to map ISO3 to the 'SOVEREIGNT' or 'ADMIN' field
    % OR join directly if the shapefile has an ISO column (it usually does!)
    % Natural Earth files typically use 'ISO_A3' for ISO3 codes.
    worldData = outerjoin(world, myData, 'LeftKey', "ISO_A3", 'RightKey', "ISO3");

    % 4. Plot all countries
    figure
    geoplot(worldData, 'ColorVariable', "Metric")
    colorbar
    title('Complete World Choropleth (ISO3)')
end

flag.other_analysis = 0;
if flag.other_analysis
    %% EIGENVALUES
    eigVs = nan(k,k,p, n_countries);
    eigDs = nan(k,  p, n_countries);
    for i=1:n_countries
        [ eigVs(:,:,1,i),eigDi ] = eig(A_KFs(:,:,1,i));
        eigDi                    = diag(eigDi);
        % % Separate real and imaginary parts of the eigenvectors and eigenvalues (NEED TO DEBUG)
        % k_eq = 0; % equation ranging from 1 to k
        % while k_eq<k
        %     k_eq = k_eq + 1;                               % next equation
        %     if abs(imag(eigDi(k_eq))) > 0.001              % start of a conjugate pair
        %         eigDi(k_eq  )       = real(eigDi(k_eq+1));
        %         eigDi(k_eq+1)       = imag(eigDi(k_eq  ));
        %         eigVs(:,k_eq,  1,i) = real(eigVs(:,k_eq+1,1,i));
        %         eigVs(:,k_eq+1,1,i) = imag(eigVs(:,k_eq,  1,i));
        %         k_eq                = k_eq + 1;            % skip the conjugate pair
        %     end
        % end
        eigDs(:,1,i)             = eigDi;
    end

    %  Plots
    fig = figure(131); fig.Name = 'A_KF eigs'; tiledlayout('flow'), clf
    % clims = [min(A_KFs(:))*0.5 max(A_KFs(:))*0.9];

    for i=1:n_countries
        nexttile
        imagesc(real([ eigVs(:,:,1,i)
            eigDs(:,  1,i)' ])), colorbar, title(countries{i})
    end

    %% SINGLE COUNTRY ANALYSIS
    country  = 'BRA'; % ENSO better with COVID, moisture is complex
    % country  = 'MEX'; % ENSO NG without COVID, good with COVID
    country  = 'COL'; % longer duration than Brazil, ENSO good with COVID
    % country  = 'IND'; % higher inflation?
    country  = 'PHL'; % deflation NG, PER, ZAF
    % country  = 'THA'; % NG

    [ countries,Y_names,X_names,k,q,n,nb,T,A_KF,B_KF,y_diff_KF,Beta,beta_filt,y,x,...
        DY,DX,dates,Sigma_u,P0,IRF_Y ] = ...
        f_results(csv_file,country,p,idx_XY,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,...
        flag.plot_irf,fig_no_irf,0,Beta0);

    %% ENDOGENOUS IRF (r=0, i=1)
    j_resp = 2;
    j_inp  = 3;

    [ irf_KF,IRF_KF ]       = f_irf_Y(hor,A_KF,DY);

    fig = figure(fig_no_irf);
    subplot(3,2,1)
    ts     = 1:hor+1;
    plot(ts,IRF_Y(:,j_resp,j_inp),'b-',ts,IRF_KF(:,j_resp,j_inp),'r--', ...
        'LineWidth',1.5)
    xlabel('time'), ylabel('irf'), title('Endogenous shock')

    %% PRINT IRF
    % fprintf('IRF:\n')
    % disp(squeeze(irf(   2,:,:)))
    % disp(squeeze(irf_KF(2,:,:)))
    % fprintf('IRF_raw:\n')
    % disp(squeeze(IRF(   2,:,:)))
    % disp(squeeze(IRF_KF(2,:,:)))
    % fprintf('Compare with res_var.irf(12).irfs[1,:,:] transpose\n')
    % fprintf('  MATLAB rows are equations\n')

    %% ROLLING FORECAST
    t1 = 10;        % start estimation based on data from 1:t1
    t1 = max(t1,k+q+3);
    H  = 1;         % predict y(t+H) based on x,y(1:t), null model is better for H = 1
    tc = H+8 + t1;  % start time to make correlation and plot comparisons
    tc = 30;

    [ yhat,    Yhat,    B_enso_VAR ] = rolling_forecast(t1,p,H,y,x,DY,0);
    [ yhat_KF, Yhat_KF, B_enso_KF  ] = rolling_forecast(t1,p,H,y,x,DY,1,[],[],[],[],beta_filt);
    [ yhat_KF0,Yhat_KF0,B_enso_KF0 ] = rolling_forecast(t1,p,H,y,x,DY,2,shrink,eps_scale,R_mat,lambda);

    ylims = 0.7;
    fig = figure(112); fig.Name = 'ENSO coeff';
    for k_idx=1:k
        subplot(4,3,3*(k_idx-1)+1), plot(squeeze(B_enso_VAR(:,k_idx,:))), ylim([-1 1]*ylims)
        subplot(4,3,3*(k_idx-1)+2), plot(squeeze(B_enso_KF( :,k_idx,:))), ylim([-1 1]*ylims)
        subplot(4,3,3*(k_idx-1)+3), plot(squeeze(B_enso_KF0(:,k_idx,:))), ylim([-1 1]*ylims)
    end

    fig = figure(102); fig.Name = 'Rolling estimates'; clf
    for k_idx=1:k
        yk       = y(       tc:end,k_idx  );
        yhatk    = yhat(    tc:end,k_idx,H);
        yhatkKF  = yhat_KF( tc:end,k_idx,H);
        yhatkKF0 = yhat_KF0(tc:end,k_idx,H);
        subplot(k,7,7*(k_idx-1)+(1:3))
        plot(tc:T,yhatk,   'r-',  tc:T,yhatkKF, 'r--', ...
            tc:T,yhatkKF0,'k:',  tc:T,yk,      'b-','LineWidth',1.5)
        xlabel('time'), ylabel('Y'), title('Rolling estimates')
        subplot(k,7,7*(k_idx-1)+4)
        plot(yk,yhatk,   'bo'), title(sprintf('%.2f',corr(yk,yhatk,   'Rows','pairwise')))
        subplot(k,7,7*(k_idx-1)+5)
        plot(yk,yhatkKF, 'bo'), title(sprintf('%.2f',corr(yk,yhatkKF, 'Rows','pairwise')))
        subplot(k,7,7*(k_idx-1)+6)
        plot(yk,yhatkKF0,'bo'), title(sprintf('%.2f',corr(yk,yhatkKF0,'Rows','pairwise')))
        subplot(k,7,7*(k_idx-1)+7)
        plot(yk(1:end-H),yk(H+1:end),'bo')
        title(            sprintf('%.2f',corr(yk(1:end-H),yk(1+H:end),'Rows','pairwise')))
    end

    %% PLOT OF ERRORS
    flag.VAR= 0; % see rolling_forecast()
    n_lambda= 10; n_R = 7;
    lambdas = linspace(0.9,1.0,n_lambda);
    Rs      = logspace(-1,3,n_R);          % from 10^-1 to 10^3
    Ls      = nan(n_R,n_lambda);
    for i=1:n_R
        for j=1:n_lambda
            Ls(i,j) = forecast_loss([lambdas(j) Rs(i)], ...
                x,y,Beta,P0,Sigma_u,p,t1,tc,H,DY,flag.VAR,shrink,eps_scale);
        end
    end

    fig = figure(110); fig.Name = 'loss contours';
    contourf(lambdas,log10(Rs),Ls), xlabel('lambda'), ylabel('R'), title('Loss'), colorbar

    keyboard

    %% OPTIMIZED KF WRT HYPERPARAMETERS
    theta0    = [lambda R_mat];
    L0        = forecast_loss(theta0,   x,y,Beta,P0,Sigma_u,p,t1,tc,H,DY,flag.VAR,shrink,eps_scale);

    lb = [ 0.90; 1e-3];
    ub = [1.000; 1000];

    theta_opt = fmincon(@(th) ...
        forecast_loss(th,       x,y,Beta,P0,Sigma_u,p,t1,tc,H,DY,flag.VAR,shrink,eps_scale), ...
        theta0, [], [], [], [], lb, ub);
    L_opt     = forecast_loss(theta_opt,x,y,Beta,P0,Sigma_u,p,t1,tc,H,DY,flag.VAR,shrink,eps_scale);

    fprintf('theta_opt = %.5f, R_mat_opt = %.5f\n',theta_opt)

    %% CHANGES
    %  res_var.params = [B A] not [A B]
end

end


function f_setup_irf()

    function COMM_ENSO = load_data()
        COMM_ENSO = [
            0.012518393	-0.383333333
            -3.531976854	0.63
            -6.428428169	1.86
            -4.190773827	2.3533333333333335
            -11.2545881	1.9166666666666667
            -15.05741881	0.32
            -14.49854857	-1.176666667
            -10.93929129	-1.536666667
            -12.28762789	-1.32
            -14.10494994	-1.15
            -10.29634573	-1.16
            -9.691189661	-1.516666667
            -5.349097955	-1.433333333
            -0.846084827	-0.84
            -3.143075422	-0.56
            -4.610227599	-0.803333333
            -5.31631759	-0.613333333
            -6.112729226	-0.34
            1.0339407737270712	-0.096666667
            -4.016666652	-0.306666667
            -1.266395435	-0.06
            3.879041627135216	0.35
            9.461572028475974	0.9
            18.44609949080096	1.3633333333333333
            16.548362860083344	0.5333333333333333
            9.712808533264504	-0.346666667
            0.9943020680132086	0.2466666666666666
            7.756476818407165	0.3933333333333333
            13.668327797214609	0.2066666666666666
            16.96910043552007	0.086666667
            11.074458791297117	0.6866666666666666
            -0.203052312	0.71
            -1.374241938	0.49
            -0.133926493	0.1999999999999999
            4.114000174341648	-0.106666667
            7.611138684009111	-0.523333333
            8.426652969489123	-0.763333333
            8.719151919973033	-0.136666667
            10.491697988689852	0.3066666666666666
            13.460313705021454	0.9433333333333334
            12.29689581313953	0.2166666666666666
            14.526734161928731	-0.38
            21.51019201636757	-0.806666667
            26.466195406427406	-1.503333333
            40.401411640782435	-1.52
            47.967850613254505	-0.836666667
            30.39958548875623	-0.226666667
            -7.262716425	-0.556666667
            -19.4219188	-0.79
            -21.7164213	0.006666667
            -16.73442853	0.5733333333333334
            12.419396598596457	1.3633333333333333
            11.280267437922298	1.2233333333333334
            3.267571456	-0.176666667
            11.03778102864812	-1.353333333
            20.87399313239959	-1.643333333
            34.60221661901861	-1.193333333
            34.45645729295062	-0.556666667
            22.771621822382883	-0.626666667
            -1.009430998	-1.1
            -10.70164144	-0.716666667
            -9.657391713	-0.266666667
            -3.494463151	0.3666666666666667
            0.8434370988539053	0.056666667
            -2.604271827	-0.433333333
            -5.103413035	-0.356666667
            -12.60544905	-0.316666667
            -9.570761181	-0.17
            -5.443118241	-0.463333333
            -0.645105723	0.21
            -2.763845557	0.066666667
            -5.519386576	0.6333333333333333
            -11.7797426	0.4666666666666666
            -16.00423932	0.9333333333333332
            -14.09337828	1.8666666666666665
            -13.11082349	2.58
            -9.516285053	2.15
            0.4750604785887402	0.3933333333333333
            3.5599635653214445	-0.546666667
            4.283038097	-0.666666667
            6.890029521	-0.16
            -2.606557678	0.31
            -3.338054734	-0.113333333
            -2.73522865	-0.836666667
            -0.254843988	-0.853333333
            3.957335287815855	-0.223333333
            -2.051276431	0.2266666666666666
            -2.980654935	0.8966666666666666
            -6.058263016	0.7233333333333333
            -8.617493816	0.54
            -3.734785026	0.1399999999999999
            2.6176237251402723	0.5066666666666667
            2.7574107425201166	0.4833333333333333
            -1.422657999	-0.083333333
            6.391959539686276	-0.573333333
            11.144988441124216	-1.276666667
            20.423730095416136	-0.933333333
            32.65052663559403	-0.486666667
            25.324213702092703	-0.49
            17.52504312229992	-0.98
            18.27179254984188	-0.933333333
            18.231503908123024	-0.99
            5.772361643800017	-0.91
            1.3054645316920244	-0.913333333
            -7.907915609	-0.43
            -13.04914184	0.48
            -4.418507673	1.3233333333333337
            -2.035045881	1.9233333333333331
            -0.897610842	1.4866666666666666
            4.356109438230349	0.3899999999999999
            3.4494435487888664	-0.113333333
            7.297340744108083	-0.373333333    ];
    end

COMM_ENSO  = load_data();                    % raw data
% COMM_ENSO  = COMM_ENSO(51:end,:);            % after 2010
COMM_ENSO  = COMM_ENSO - mean(COMM_ENSO);    % standardize
stds       = std(COMM_ENSO);

COMM_ENSO  = COMM_ENSO./std(COMM_ENSO);

COMM_ENSO1 = [COMM_ENSO(2:end,1) COMM_ENSO(1:end-1,2) COMM_ENSO(2:end,2)]; % y_t x_t+1, x_t

T   = length(COMM_ENSO)

mdl = estimate_VARX(COMM_ENSO( :,2), [],   1)              % x_t ~ x_t-1

mdl = estimate_VARX(COMM_ENSO( :,1), COMM_ENSO( :,2),   1) % y_t ~ y_t-1 + x_t-1

mdl = estimate_VARX(COMM_ENSO1(:,1), COMM_ENSO1(:,2:3), 1) % y_t ~ y_t-1 + x_t + x_t-1

mdl = estimate_VARX(COMM_ENSO1(:,1), COMM_ENSO1(:,2),   1) % y_t ~ y_t-1 + x_t

mdl = estimate_VARX(COMM_ENSO1(:,1), COMM_ENSO1(:,2),   2) % y_t ~ y_t-1 + x_t

mdl = estimate_VARX(COMM_ENSO1(:,1), [ones(T-1,1) COMM_ENSO1(:,2) COMM_ENSO1(:,2).^2],   0) % y_t ~ x_t + x_t^2

ts = 1997:0.25:2026;
ts = ts(1:T);
figure(93); tiledlayout('flow'),
nexttile, plot(ts,COMM_ENSO), yline(0), legend('COMM','ENSO'), grid on, title('ENSO, Agr COMM')
nexttile, plot(COMM_ENSO(:,2),COMM_ENSO(:,1),'bo')
xlabel('ENSO'), ylabel('COMM'), yline(0), xline(0), title('all data')
nexttile, plot(COMM_ENSO( 1:50, 2),COMM_ENSO( 1:50, 1),'bo')
xlabel('ENSO'), ylabel('COMM'), yline(0), xline(0), title('before 2010')
cs = COMM_ENSO(51:end,2);
es = COMM_ENSO(51:end,1);

mdl = estimate_VARX(cs, [], 1)
mdl = estimate_VARX(es, [], 1)

nexttile, plot(cs,es,'bo')
xlabel('ENSO'), ylabel('COMM'), yline(0), xline(0), title('after 2010 with exponential fit')
Es     = sort(es);
hold on,  plot(Es,10/stds(1)*(exp(-1.0*Es)-1),'r-','LineWidth',1), hold off

% std(COMM) = 13.2

% keyboard

end


function [ idx,n ] = analysis_A(As,hor,lim)

% [V,D,W] = eig(A), A*V = V*D, V'*W = D1, diag(V'*V) = diag(W'*W) = 1
% 
% y0 = V*a0, y1 = A*y0 = A*V*a0 = V*D*a0, yn = V*D^n*a0
% y0 = B*b0 = V*a0 => a0 = 1/D1*W'*V*a0 = 1/D1*W'*B*b0

As    = squeeze(As(:,:,1,:));        % (k,k,p,n_countries) -> (k,k,n_countries)
n_countries = size(As,3);
for i=1:n_countries
    A_eig  = eig(As(:,:,i));
    idx(i) = max(A_eig.^hor) < lim;  % response is less than limit at horizon
end

n   = sum(idx);

end


function plot_Beta(fig_no,fig_name,Betas,A_KFs,B_KFs,DYs,clims,countries,Beta_sim)

return

fig = figure(fig_no); fig.Name = fig_name; tiledlayout('flow'), clf

for i=1:numel(countries)
    if isempty(Betas)
        if isempty(DYs)
            Beta_i = [A_KFs(:,:,1,i)' ; B_KFs(:,:,i)'];
        else
            Beta_i = [inv(DYs{i})*A_KFs(:,:,1,i)' ; B_KFs(:,:,i)']*DYs{i};
        end
    else
        Beta_i = Betas(:,:,i);
        if ~isempty(DYs)
            [n,k]            = size(Betas(:,:,i));
            invDYXi          = eye(n);
            invDYXi(1:k,1:k) = inv(DYs{i});
            Beta_i           = invDYXi*Beta_i*DYs{i};
        end
    end
    nexttile, imagesc(Beta_i), colorbar
    title(countries{i}), clim(clims)
end

if ~isempty(Beta_sim)
    nexttile, imagesc(Beta_sim), colorbar, title('Simulated Beta'), clim(clims)
end

end


function [ countries,Y_names,X_names,k,q,n,nb,T,A_KF,B_KF,y_diff_KF,Beta,beta_filt,y,x,...
    DY,DX,dates,Sigma_u,P0,IRF_Y,a_filt ] = ...
    f_results(csv_file,country,p,idx_XY,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,...
    flag_plot,flag_plot_irf,fig_no_irf,Beta0_in,V_stacked,n_SVD,beta_mean,beta_sigma,flag_plot_pdf)

%% DATA
[ countries, Y_names, X_names, Y, X, dates ] = load_country_VAR_data( ...
    csv_file, country, idx_XY, flag_plot);

X_names = string(X_names); % only once
Y_names = string(Y_names);
k     = numel(Y_names);
q     = numel(X_names);

%% INITIALIZE
n  = p*k + q;          % regressors per equation
nb = k * n;            % total state dimension

if isempty(country)
    return
end

%% DATA PREP
%  Remove COVID
T       = size(Y,1);
T_covid = T - T_covid_end;
Ts      = 1:T_covid-1;
X       = X(Ts,:);
Y       = Y(Ts,:);
dates   = dates(Ts,:);
T       = length(Ts);                               % get new T

if T <= p + n
    fprintf('Not enough observations for %s: T must be > p + n\n', country);
    return
end

Y_std   = std(Y, 0, 1);  y = (Y-mean(Y)) ./ Y_std;  % standardize
X_std   = std(X, 0, 1);  x = (X-mean(X)) ./ X_std;
DY      = diag(Y_std);
DX      = diag(X_std);

%% VARX
res     = estimate_VARX(y, x, p);
A       = res.A;            % k × k × p
B       = res.B;            % k × q

Sigma_u = res.Sigma_u; Z = res.Z; ZZ = res.ZZ; Beta = res.Beta;

%% IRF
[ irf_y,IRF_Y ] = f_irf_Y(hor,A,DY);

%% P0      = VAR[Beta]
%  R_mat   = VAR[innovation]
%  sigma_u = VAR[epsilon] = Q,  Beta(t+1) = Beta(t) + epsilon
try
    if isnan(beta_sigma(1)), error('dummy message'); end  % proceed to f_P0() if isnan  
    P0 = diag(beta_sigma);
catch ME
    P0 = f_P0(ZZ,Sigma_u,p,shrink,eps_scale);
end

if isempty(Beta0_in)
    Beta0 = Beta;          % use Beta from VARX, otherwise use Beta0 from args
else
    Beta0 = Beta0_in;
end

%  If V_stacked is not passed, need to set default empty arrays
try V_stacked; catch, V_stacked = []; n_SVD = []; beta_mean = []; end

%  Full-sized Beta0,P0,beta_mean
%             beta_filt
%  SVD-sized  a_filt (is empty if no SVD)
[ A_KF,B_KF,beta_filt,a_filt ] = f_KF(x,y,Beta0,P0,Sigma_u,R_mat,lambda,p,...
    V_stacked,n_SVD,beta_mean);

%% EXOGENOUS IMPULSE
%  MEX: j_ex = 1, shock_size = -1: (only 1 x) ENSO good, moist CPI goes down
%  IDN: j_ex = 1, shock_size = -1: NG (both ENSO and moist gives opposite effects)
%  ZAF: ENSO NG, moist good
%  0-dimensional arrays if there is no X, j_ex is ignored

[ y_diff, y_diff_KF ] = f_irf_X(j_ex,shock_size,hor,A,A_KF,B,B_KF,flag_plot_irf,fig_no_irf);

if p==0
    A_KF = zeros(k);
end

flag_EM_calc = 1;
% if flag_EM_calc & isempty(n_SVD)  % cannot do SVD yet
if flag_EM_calc                   % EM
    P00            = P0;          % from f_P0
    Beta00         = Beta0;       % simple initial value (scaled identity)
    ts             = dates.quarter;
    dts            = ts(2:end);

    flag_EM        = 2;           % 1-approximate EM, 2-EM with cross-covariance
    P0             = 0.1*P00;
    Beta0          = Beta00;
    R_scale0       = 0.25;
    Q              = 0.1*P00;

    % beta_filt_test = f_KF_basic(x,y,Beta0,P0,Sigma_u,R_scale,lambda,p);
    % [beta_filt_test,beta_smooth,P_filt0,P_smooth0,v_all,S_all0,innov_score, ...
    %       dbeta_smooth,smooth_filt_gap] = ...
    %     f_KF_smoother(x,y,Beta0,P0,Sigma_u,R_scale,lambda,p);

    if flag_EM
        n_iter = 10;
        if flag_EM==1
            % Need to modify to accommodate V_stacked,...
            % [Beta0,P0,Q,R_scale,loglik_hist] = ...
            %     f_EM_KF_Q(x,y,Beta0,P0,Q,Sigma_u,R_scale0,p,n_iter);
        else
            R_scale = R_scale0 * Sigma_u;

            % Full sized Beta0,P0,Q
            % First time      - no SVD
            % Subsequent time -    SVD
            [Beta0,P0,Q,R_scale,loglik_hist,Qs,R_scales] = ...
                f_EM_KF_Q2(x,y,Beta0,P0,Q,Sigma_u,R_scale,p,n_iter,...
                V_stacked,n_SVD,beta_mean);
        end
    else
        loglik_hist = nan(T,1);
    end

    % [beta_filt_test,beta_smooth,P_filt0,P_smooth0, ...
    %       v_all,S_all0,innov_score, ...
    %       dbeta_smooth,smooth_filt_gap] = ...

    % Full sized Beta0,P0,Q
    %            beta_filt_test,beta_smooth,P_filt0,P_smooth0
    % f_KF_smoother_Q() is also called inside f_EM_KF_Q2()
    flag_V = 0;  % full analysis
    [beta_filt_test,beta_smooth,P_filt0,P_smooth0, J_all,P_lag, v_all,S_all0,innov_score, ...
          dbeta_smooth,smooth_filt_gap] = f_KF_smoother_Q(x,y,Beta0,P0,Q,Sigma_u,R_scale,p,...
          flag_V,V_stacked,n_SVD,beta_mean);

    [ A_KF,B_KF ] = extract_AB(beta_filt_test(:,T), p, k, q);
    beta_filt     = beta_filt_test;
end

try flag_plot_pdf; catch, flag_plot_pdf = 0; end

flag_plot_detail       = 0;
flag_save_break_scores = 1;

if flag_plot_detail || flag_plot_pdf || flag_save_break_scores
    %% ANALYSIS OF STRUCTURAL BREAKS (use S also?)
    beta_jump = sqrt(sum(dbeta_smooth   .^2,1)); % (+)
    gap       = sqrt(sum(smooth_filt_gap.^2,1)); % (+)
    d_gap     = [NaN diff(gap)];                 % (+/-)
    d_gap     = abs(d_gap);                      % (+)
    innov_sm  = movmean(innov_score,4);          % (+)
    z_innov   = (innov_sm  - mean(innov_sm, 'omitnan')) / std(innov_sm, 'omitnan');
    z_jump    = (beta_jump - mean(beta_jump,'omitnan')) / std(beta_jump,'omitnan');
    z_gap     = (gap       - mean(gap,      'omitnan')) / std(gap,      'omitnan');
    z_d_gap   = (d_gap     - mean(d_gap,    'omitnan')) / std(d_gap,    'omitnan');
    break_scores = [ z_innov ; z_jump ; z_d_gap ]';
%   break_scores = [ z_innov ; z_jump ; z_gap   ]';
end

if flag_plot_detail || flag_plot_pdf
    % Asian crisis            1997–1998
    % China WTO entry         2001
    % Global financial crisis 2008–2009
    % Commodity collapse      2014–2016
    % COVID                   2020

    Country  = f_country_name(country);

    % Put in vector form for plotting the diagonal elements
    P_filt   = reshape(P_filt0,  nb^2,[]);
    P_smooth = reshape(P_smooth0,nb^2,[]);
    S_all    = reshape(S_all0,k^2,[]);

    ns = 1:nb;

    eigs_A    = nan(k,T);
    B_t       = nan(q*k,T);
    for t=1:T
        beta_t      = reshape(beta_smooth(:,t),n,k);
        eigs_A(:,t) = sort(abs(eig(beta_t(1:k,:))));
        B_t(   :,t) = reshape(beta_t(k+1:end,:),[],1);
    end

end

if flag_save_break_scores
    Tbl = table(ts,repmat(country,numel(ts),1),...
        break_scores(:,1),break_scores(:,2),break_scores(:,3),...
        'VariableNames',{'date','country','x1','x2','x3'});
    writetable(Tbl,'break_scores.xlsx','FileType','spreadsheet','WriteMode','append')
end

if flag_plot_detail
    fig = figure(99); fig.Name = 'test'; tiledlayout('flow')
    nexttile, imagesc(beta_filt_test),         xlabel('t'), ylabel('beta'), title('KF')
    nexttile, imagesc(beta_smooth),            xlabel('t'), ylabel('beta'), title('smoother')
    nexttile, plot(ts,beta_filt_test'),        xlabel('t'), ylabel('beta'), title('KF')
    nexttile, plot(ts,beta_smooth'),           xlabel('t'), ylabel('beta'), title('smoother')
    nexttile, imagesc(P_filt(  1:nb+1:end,:)), xlabel('t'), ylabel('P'),    title('KF')
    nexttile, imagesc(P_smooth(1:nb+1:end,:)), xlabel('t'), ylabel('P'),    title('smoother')
    nexttile, plot(ts,v_all'),                 xlabel('t'), ylabel('v'),    title('v')
    nexttile, plot(ts,S_all'),                 xlabel('t'), ylabel('S'),    title('S')
%   nexttile, plot(ts,innov_score),            xlabel('t'), ylabel('score'),title('innovation score')
    nexttile, plot(ts,eigs_A),                 xlabel('t'), ylabel('eigs'), title('eigs of A')
    nexttile, if ~isempty(B_t), plot(ts,B_t),  xlabel('t'), ylabel('beta'), title('B'), end
    nexttile, plot(ts,break_scores),           xlabel('t'), ylabel('score'),title('break score')
    grid on
    hold on,  plot(ts,mean(break_scores,2),'k--','LineWidth',2), hold off
    nexttile, imagesc(dbeta_smooth),           xlabel('t'), ylabel('Dbeta'),title('D smoother')
    nexttile, plot(ts,dbeta_smooth'),          xlabel('t'), ylabel('Beta'), title('D smoother')
%   nexttile, imagesc(smooth_filt_gap),        xlabel('t'), ylabel('Dbeta'),title('Smoother-filt')
    nexttile, plot(ts,smooth_filt_gap'),       xlabel('t'), ylabel('Dbeta'),title('Smoother-filt')
    nexttile, plot(dts,diff(smooth_filt_gap')),xlabel('t'), ylabel('Dbeta'),title('D Smoother-filt')
%   nexttile, plot(2:n_iter,loglik_hist(2:n_iter)), xlabel('iteration'), ylabel('llik'), title('Log likelihodd')
    if flag_EM==2
%       nexttile, plot(R_scales),      xlabel('iteration'), ylabel('R'),    title('R scale')
        nexttile, plot(Qs'),           xlabel('iteration'), ylabel('Q'),    title('Q')
    end
end

if flag_plot_pdf
    fig1 = figure('Visible','off','Color','w','Name','test1','MenuBar', 'none', 'ToolBar', 'none'); t1 = tiledlayout(2,2);
    ax = nexttile(1); imagesc(ts,ns,beta_filt_test),   xlabel('Year'), ylabel('Beta'), grid on, title('Kalman Filter')
    ax.Toolbar = [];
    ax = nexttile(2); imagesc(ts,ns,beta_smooth),      xlabel('Year'), ylabel('Beta'), grid on, title('Kalman Smoother')
    ax.Toolbar = [];
    ax = nexttile(3); plot(ts,beta_filt_test'),        xlabel('Year'), ylabel('Beta'), grid on, title('Kalman Filter')
    ax.Toolbar = [];
    ax = nexttile(4); plot(ts,beta_smooth'),           xlabel('Year'), ylabel('Beta'), grid on, title('Kalman Smoother')
    ax.Toolbar = [];
    title(t1, Country, 'FontSize', 24, 'FontWeight', 'bold')

    fig2 = figure('Visible','off','Color','w','Name','test2','MenuBar', 'none', 'ToolBar', 'none'); t2 = tiledlayout(2,2);
    ax = nexttile(1); imagesc(ts,ns,beta_filt_test),   xlabel('Year'), ylabel('Beta'), grid on, title('Kalman Filter')
    ax.Toolbar = [];
    ax = nexttile(2); imagesc(ts,ns,beta_smooth),      xlabel('Year'), ylabel('Beta'), grid on, title('Kalman Smoother')
    ax.Toolbar = [];
    ax = nexttile(4); plot(ts,break_scores,'LineWidth',1.3),           xlabel('Year'), ylabel('Score'),grid on, title('Structural Break Scores')
    grid on,     yline(0)
    ax.Toolbar = [];
    % hold on,     plot(ts,mean(break_scores,2),'k:','LineWidth',3), hold off
    legend('Innovation','Change in beta coefficients', ...
        'Difference between Kalman filter & smoother','Location','Best')
    title(t2, Country, 'FontSize', 24, 'FontWeight', 'bold')

    if flag_plot_pdf==1
        exportgraphics(fig1,'Beta_over_time.pdf',        'BackgroundColor','w');
        exportgraphics(fig2,'Beta_structural_breaks.pdf','BackgroundColor','w');
    else
        exportgraphics(fig1,'Beta_over_time.pdf',        'BackgroundColor','w','Append',true);
        exportgraphics(fig2,'Beta_structural_breaks.pdf','BackgroundColor','w','Append',true);
    end
end

end


function Country = f_country_name(country)

% Define the ISO3 to IMF Name mapping as a MATLAB dictionary
ISO3_TO_IMF_NAME_FULL = dictionary(...
    ["AFG", "ALB", "DZA", "AND", "AGO", "ATG", "ARG", "ARM", "AUS", "AUT", ...
     "AZE", "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BTN", ...
     "BOL", "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "BDI", "CPV", "KHM", ...
     "CMR", "CAN", "CAF", "TCD", "CHL", "CHN", "COL", "COM", "COG", "COD", ...
     "CRI", "CIV", "HRV", "CUB", "CYP", "CZE", "DNK", "DJI", "DMA", "DOM", ...
     "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH", "FJI", "FIN", ...
     "FRA", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN", ...
     "GNB", "GUY", "HTI", "HND", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ", ...
     "IRL", "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "KWT", ...
     "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX", ...
     "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MRT", "MUS", "MEX", ...
     "FSM", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NRU", ...
     "NPL", "NLD", "NZL", "NIC", "NER", "NGA", "PRK", "MKD", "NOR", "OMN", ...
     "PAK", "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "QAT", ...
     "ROU", "RUS", "RWA", "KNA", "LCA", "VCT", "WSM", "SMR", "STP", "SAU", ...
     "SEN", "SRB", "SYC", "SLE", "SGP", "SVK", "SVN", "SLB", "SOM", "ZAF", ...
     "KOR", "SSD", "ESP", "LKA", "SDN", "SUR", "SWE", "CHE", "SYR", "TJK", ...
     "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV", ...
     "UGA", "UKR", "ARE", "GBR", "USA", "URY", "UZB", "VUT", "VEN", "VNM", ...
     "YEM", "ZMB", "ZWE"], ...
    ["Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", ...
     "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", ...
     "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei Darussalam", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", ...
     "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo, Republic of", "Congo, Democratic Republic of the", ...
     "Costa Rica", "Côte d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", ...
     "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", ...
     "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", ...
     "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", ...
     "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", ...
     "Kyrgyz Republic", "Lao PDR", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", ...
     "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", ...
     "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Name", "Nauru", ...
     "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Korea, North", "North Macedonia", "Norway", "Oman", ...
     "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qat", ...
     "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", ...
     "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", ...
     "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", ...
     "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", ...
     "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", ...
     "Yemen", "Zambia", "Zimbabwe"]);

Country = ISO3_TO_IMF_NAME_FULL(country);

end


function [Beta0,P0,Q,R_scale,loglik_hist,Qs,R_scales] = f_EM_KF_Q2(...
    x,y,Beta0_full,P0_full,Q0_full,Sigma_u,R_scale,p,n_iter,V,n_SVD,beta_mean0_full)

%% CALLED: f_results() during EM calculations
%  First time      - use full
%  Subsequent time - use SVD

[ T,k ]   = size(y);
[ ~,q ]   = size(x);

%% V
try
    assert(n_SVD>0);                  % error if n_SVD = 0 or was not passed
    flag_SVD  = true;
catch
    flag_SVD  = false;
end

if flag_SVD
    nb        = n_SVD;
    V         = V(:,1:n_SVD);              % take subspace
    beta_pred = V'*Beta0_full(:);          % use V subspace and convert back to whole space later
    beta_mean = V'*beta_mean0_full(:);     % V subspace (beta_mean), whole space (beta_mean0)
    P_pred    = V'*P0_full*V;
    Q         = V'*Q0_full*V;
    flag_V    = 2;                         % don't do this again in f_KF_smoothr_Q()
else
    n         = k*p + q;
    nb        = n*k;
    beta_pred = Beta0_full(:);
    beta_mean = 0;
    P_pred    = P0_full;                   % your diagonal prior
    Q         = Q0_full;
    flag_V    = 0;                         % full analysis
end

%% ORIGINAL
% n         = k*p + q;
% nb        = n*k;

loglik_hist = nan(1, n_iter);
R_scales    = nan(1, n_iter);
Qs          = nan(nb,n_iter);

flag_constraint  = 0; % 1-make P0, Q diagonal and use R = R_scale*Sigma_u
flag_Q_empirical = 1; % 1-use a simple approximation for Q
flag_iter_stop_Q = 1; % iteration number to stop updating Q

%% IF POSSIBLE USE V-COORDINATES IN THE INNER LOOP
for iter = 1:n_iter
    % --- E-step
    % Full sized Beta0,P0,Q0
    %            beta_filt_test,beta_smooth,P_filt,P_smooth
    % f_KF_smoother_Q() is also in the calling function f_results()    
    [beta_filt,beta_smooth,P_filt,P_smooth,J_all,P_lag,v_all,S_all,innov_score, ...
     dbeta_smooth,smooth_filt_gap] = ...
        f_KF_smoother_Q(x,y,beta_pred,P_pred,Q, Sigma_u,R_scale,p,flag_V,V,n_SVD,beta_mean);
%       f_KF_smoother_Q(x,y,beta_pred,P0,    Q0,Sigma_u,R_scale,p,       V,n_SVD,beta_mean);

    % --- Log-likelihood
    loglik = 0;
    for t = 2:T
        S = S_all(:,:,t);
        v = v_all(:,t);
        loglik = loglik - 0.5 * ( ...
            k*log(2*pi) + log(det(S)) + v' * (S \ v) );
    end
    loglik_hist(iter) = loglik;

    % =========================================================
    % M-step
    % =========================================================

    % --- Update Beta0
    Beta0 = beta_smooth(:,1);

    % --- Update P0
    P0 = P_smooth(:,:,1);
    if flag_constraint
        P0 = diag(diag(P0));    % optional diagonal restriction
    end

    % --- Precompute M_t = E[beta_t beta_t' | Y]
    M = zeros(nb,nb,T);
    for t = 1:T
        bt = beta_smooth(:,t);
        M(:,:,t) = P_smooth(:,:,t) + bt*bt';
    end

    % --- Update Q
    if iter<=flag_iter_stop_Q
        if flag_Q_empirical
            % use the variability of beta_smooth over time
            d_beta = diff(beta_smooth,1,2);
            Q      = cov(d_beta');
        else
            % EM algorithm expressions
            Q_new = zeros(nb,nb);
            for t = 2:T
                bt   = beta_smooth(:,t);
                btm1 = beta_smooth(:,t-1);

                M_lag = P_lag(:,:,t) + bt*btm1';

                Q_new = Q_new + M(:,:,t) + M(:,:,t-1) - M_lag - M_lag';
            end
            Q = Q_new / (T-1);
        end

        if flag_constraint
            Q = diag(diag(Q));      % optional diagonal restriction
        end
    end

    % --- Update R
    R_new = zeros(k,k);
    count = 0;
    for t = 2:T
        H  = build_Zt(y, x, t, p);
        if flag_SVD
            H = H * V;                   % (k,n_SVD)
        end
        yt = y(t,:)';
        bt = beta_smooth(:,t);

        vs = yt - H * bt;

        R_new = R_new + (vs*vs') + H * P_smooth(:,:,t) * H';
        count = count + 1;
    end
    R_new = R_new / count;

    % Keep your original parameterization R = R_scale * Sigma_u
    R_scale_scalar = trace(R_new) / trace(Sigma_u);
    if flag_constraint
        R_scale = R_scale_scalar;
    else
        R_scale = R_new;
    end

    R_scales(iter) = R_scale_scalar;
    Qs(:,iter)     = diag(Q);
end

% Beta0,P0,Q
if flag_SVD
    Beta0 = V *Beta0;              % put in V subspace, beta = V*V'*beta = V*a
    P0    = V*P0*V';
    Q     = V*Q*V';                % Q(SVD) = V'*Q *V
    % beta_filt = beta_filt + beta_mean0(:);
end

return

%% COMPARE Q WITH beta_smooth (not necessary it flag_Q_empirical=1)
d_beta = diff(beta_smooth,1,2);

fig = figure(90); fig.Name = 'Check Q'; tiledlayout('flow')
nexttile, imagesc(cov(d_beta')), colorbar, title('cov d beta')
nexttile, imagesc(Q),            colorbar, title('Q')
nexttile, plot(diag(cov(d_beta'))),        title('diag cov d beta')
nexttile, plot(diag(Q)),                   title('diag cov d beta')

end


function [Beta0,P0,Q,R_scale,loglik_hist] = f_EM_KF_Q_not_updated(x,y,Beta0,P0,Q,Sigma_u,R_scale,p,n_iter)

loglik_hist = nan(1,n_iter);

for iter = 1:n_iter
    % --- E-step
    [beta_filt,beta_smooth,P_filt,P_smooth, ...
     v_all,S_all,innov_score, ...
     dbeta_smooth,smooth_filt_gap] = ...
        f_KF_smoother_Q_not_updated(x,y,Beta0,P0,Q,Sigma_u,R_scale,p);

    % --- compute log-likelihood
    [~,k] = size(y);
    loglik = 0;
    for t = 2:size(y,1)
        S = S_all(:,:,t);
        v = v_all(:,t);
        loglik = loglik - 0.5 * ( ...
            k*log(2*pi) + log(det(S)) + v' * (S \ v) );
    end
    loglik_hist(iter) = loglik;

    % --- M-step: update Beta0
    Beta0 = beta_smooth(:,1);

    % --- M-step: update P0
    P0 = diag(diag(P_smooth(:,:,1)));   % keep diagonal if you want stability

    % --- M-step: update Q
    Q_new = zeros(size(Q));
    count = 0;
    for t = 2:size(y,1)
        d = beta_smooth(:,t) - beta_smooth(:,t-1);
        Q_new = Q_new + (d*d') + P_smooth(:,:,t) + P_smooth(:,:,t-1);
        count = count + 1;
    end
    Q = Q_new / count;
    Q = diag(diag(Q));                  % diagonal restriction for stability

    % --- M-step: update R_scale
    R_new = zeros(k,k);
    count = 0;
    for t = 2:size(y,1)
        H  = build_Zt(y, x, t, p);
        yt = y(t,:)';
        vs = yt - H * beta_smooth(:,t);

        R_new = R_new + (vs*vs') + H * P_smooth(:,:,t) * H';
        count = count + 1;
    end
    R_new = R_new / count;

    R_scale = trace(R_new) / trace(Sigma_u);
end

end


function [beta_filt,beta_smooth,P_filt,P_smooth,J_all,P_lag,v_all,S_all,innov_score, ...
          dbeta_smooth,smooth_filt_gap] = ...
    f_KF_smoother_Q(x,y,Beta0,P0,Q,Sigma_u,R_scale,p,flag_V,V,n_SVD,beta_mean0)

%  TVP-VARX Time series equations (for the special case of lag p = 1)
%  y_t+1   = A*y_t + B*x_t    + u_t
%          = beta*[y_t ; x_t] + u_t
%
%  Kalman Filter parameters
%  Beta0   = initial value for beta
%  P0      = prior variance of beta
%  Q       = state-noise covariance of beta
%  Sigma_u = variance of u_t
%  R_scale = scalar multiple used with Sigma_u for the variance of the innovation R
%  p       = maximum lag of the endogenous variables y

[ T,k ]   = size(y);
[ ~,q ]   = size(x);

%% V
if flag_V % 1-transform everything or 2-only transform H and include beta_mean
    V         = V(:,1:n_SVD);         % take subspace
    nb        = n_SVD;
    if flag_V==1
        flag_SVD  = true;
        beta_mean = V'*beta_mean0(:);     % V subspace (beta_mean), whole space (beta_mean0)
        beta_pred = V'*Beta0(:);          % use V subspace and convert back to whole space later
        P_pred    = V'*P0*V;
        Q         = V'*Q *V;
    else
        flag_SVD  = false;
        beta_mean = beta_mean0(:);
        beta_pred = Beta0(:);
        P_pred    = P0;    % your diagonal prior
    end
else
    flag_SVD  = false;
    n         = k*p + q;
    nb        = n*k;
    beta_mean = 0;
    beta_pred = Beta0(:);
    P_pred    = P0;    % your diagonal prior    
end

%% ORIGINAL
% n         = k*p + q;
% nb        = n*k;
% 
% beta_pred = Beta0(:);
% P_pred    = P0;

%% CONTINUE WITH nb
beta_filt = zeros(nb, T);
P_filt    = zeros(nb, nb, T);
I         = eye(nb);

% --- store predicted quantities for smoother
beta_pred_all = zeros(nb, T);
P_pred_all    = zeros(nb, nb, T);

% --- store smoother gain
J_all         = zeros(nb, nb, T);

% --- store lag-one smoothed covariance
P_lag         = zeros(nb, nb, T);   % P_lag(:,:,t) = Cov(beta_t, beta_{t-1} | Y), t>=2

% --- store innovation diagnostics
v_all        = zeros(   k, T);
S_all        = zeros(k, k, T);
innov_score  = nan(     1, T);

% --- Measurement noise covariance
if isscalar(R_scale)
    R = R_scale * Sigma_u;
else
    R = R_scale;
end

% --- Initialize time 1 with the prior
beta_filt(    :,1) = beta_pred; % Beta0(:);
beta_pred_all(:,1) = beta_pred; % Beta0(:);
P_filt(    :,:,1)  = P_pred;    % P0;
P_pred_all(:,:,1)  = P_pred;    % P0;

%%    Kalman filter
for t = 2:T
    % --- Prediction step
    P_pred = P_pred + Q;

    % --- Store predicted quantities
    beta_pred_all(:,t) = beta_pred;
    P_pred_all( :,:,t) = P_pred;

    % --- Measurement
    % ZVt = Zt*V
    Zt    = build_Zt(y, x, t, p);      % (k,   nb_orig), time t-1 ... t-p
    if flag_V                          % both 1 or 2
        Zt = Zt * V;                   % (k,n_SVD)
    end
    H     = Zt;

    % --- Innovation
    yt    = y(t,:)';
%   v     = yt - H * beta_pred;
    v     = yt - H * (beta_pred + beta_mean);

    % --- Innovation covariance
    S = H * P_pred * H' + R;

    % --- Store innovation diagnostics
    v_all(:,t)     = v;
    S_all(:,:,t)   = S;
    innov_score(t) = real(v' * (S \ v));

    % --- Kalman gain
    K = P_pred * H' / S;

    % --- Update
    beta_filt(:,t) = beta_pred + K * v;
    KH             = K * H;
    P_filt(:,:,t)  = (I - KH) * P_pred * (I - KH)' + K * R * K';

    % --- Prepare next iteration
    beta_pred = beta_filt(:,t);
    P_pred    = P_filt(:,:,t);
end

%%    RTS smoother
beta_smooth = zeros(nb, T);
P_smooth    = zeros(nb, nb, T);

% --- Start backward pass from final filtered estimate
beta_smooth(:,T) = beta_filt(:,T);
P_smooth( :,:,T) = P_filt( :,:,T);

for t = T-1:-1:1
    % Since F = I:
    J = P_filt(:,:,t) / P_pred_all(:,:,t+1);
    J_all(:,:,t) = J;

    % --- Smoothed state
    beta_smooth(:,t) = beta_filt(:,t) ...
        + J * (beta_smooth(:,t+1) - beta_pred_all(:,t+1));

    % --- Smoothed covariance
    P_smooth(:,:,t) = P_filt(:,:,t) ...
        + J * (P_smooth(:,:,t+1) - P_pred_all(:,:,t+1)) * J';
end

%%    Lag-one smoothed covariance for EM
% Initialization at final time
% P_lag(:,:,T) = Cov(beta_T, beta_{T-1} | Y)

P_lag(:,:,T) = (I - zeros(nb)) * J_all(:,:,T-1) * P_filt(:,:,T-1);  % overwritten below if T>2

if T >= 3
    % A practical and standard backward recursion
    % First initialize using final smoother gain
    P_lag(:,:,T) = P_smooth(:,:,T) * J_all(:,:,T-1)';

    for t = T-1:-1:2
        Jt   = J_all(:,:,t);
        Jtm1 = J_all(:,:,t-1);

        P_lag(:,:,t) = P_smooth(:,:,t) * Jtm1' ...
            + Jt * (P_lag(:,:,t+1) - P_smooth(:,:,t)) * Jtm1';
    end
end

%%    Post-smoother diagnostics
dbeta_smooth = nan(nb, T);
for t = 2:T
    dbeta_smooth(:,t) = beta_smooth(:,t) - beta_smooth(:,t-1);
end

smooth_filt_gap = beta_smooth - beta_filt;

if flag_SVD                             % only do this if we want to go back to original space
    beta_filt = V *a_filt;              % put in V subspace, beta = V*V'*beta = V*a
    beta_filt = beta_filt + beta_mean0(:);
end


end


function [beta_filt,beta_smooth,P_filt,P_smooth,v_all,S_all,innov_score, ...
          dbeta_smooth,smooth_filt_gap] = ...
    f_KF_smoother_Q_old(x,y,Beta0,P0,Q,Sigma_u,R_scale,p)

%  TVP-VARX Time series equations (for the special case of lag p = 1)
%  y_t+1   = A*y_t + B*x_t    + u_t
%          = beta*[y_t ; x_t] + u_t
%
%  Kalman Filter parameters
%  Beta0   = initial value for beta
%  P0      = prior variance of beta
%  Q       = state-noise covariance of beta
%  Sigma_u = variance of u_t
%  R_scale = scalar multiple used with Sigma_u for the variance of the innovation R
%  p       = maximum lag of the endogenous variables y

[ T,k ]   = size(y);
[ ~,q ]   = size(x);

n         = k*p + q;
nb        = n*k;

beta_pred = Beta0(:);
P_pred    = P0;                        % prior variance

beta_filt = zeros(nb, T);
P_filt    = zeros(nb, nb, T);
I         = eye(nb);

% --- store predicted quantities for smoother
beta_pred_all = zeros(nb, T);
P_pred_all    = zeros(nb, nb, T);

% --- store innovation diagnostics
v_all        = zeros(k, T);
S_all        = zeros(k, k, T);
innov_score  = nan(1, T);              % v' * inv(S) * v

% --- Measurement noise covariance
R = R_scale * Sigma_u;                 % k × k

% --- Initialize time 1 with the prior
beta_filt(    :,1) = Beta0(:);
beta_pred_all(:,1) = Beta0(:);
P_filt(    :,:,1)  = P0;
P_pred_all(:,:,1)  = P0;

%%    Kalman filter
for t = 2:T                            % start with yt at time 2
    % --- Prediction step
    % Since state transition is identity:
    % beta_pred stays equal to previous filtered beta
    P_pred = P_pred + Q;

    % --- Store predicted quantities for smoothing
    beta_pred_all(:,t) = beta_pred;
    P_pred_all( :,:,t) = P_pred;

    % --- Measurement
    H     = build_Zt(y, x, t, p);      % (k, nb), time t-1 ... t-p

    % --- Innovation
    yt    = y(t,:)';                   % (k,1), time t
    v     = yt - H * beta_pred;

    % --- Innovation covariance
    S = H * P_pred * H' + R;           % (k,k)

    % --- Store innovation diagnostics
    v_all(:,t)     = v;
    S_all(:,:,t)   = S;
    innov_score(t) = real(v' * (S \ v));

    % --- Kalman gain
    K = P_pred * H' / S;

    % --- Update
    beta_filt(:,t) = beta_pred + K * v;
    KH             = K * H;
    P_filt(:,:,t)  = (I - KH) * P_pred * (I - KH)' + K * R * K';

    % --- Prepare next iteration
    beta_pred = beta_filt(:,t);
    P_pred    = P_filt( :,:,t);
end

%%    RTS smoother
beta_smooth = zeros(nb, T);
P_smooth    = zeros(nb, nb, T);

% --- Start backward pass from final filtered estimate
beta_smooth(:,T) = beta_filt(:,T);
P_smooth( :,:,T) = P_filt( :,:,T);

for t = T-1:-1:1
    % Since state transition matrix F = I, smoother gain is:
    % J_t = P_filt(:,:,t) * inv(P_pred_all(:,:,t+1))
    J = P_filt(:,:,t) / P_pred_all(:,:,t+1);

    % --- Smoothed state
    beta_smooth(:,t) = beta_filt(:,t) + J * (beta_smooth(:,t+1) - beta_pred_all(:,t+1));

    % --- Smoothed covariance
    P_smooth(:,:,t) = P_filt(:,:,t)   + J * (P_smooth(:,:,t+1) - P_pred_all(:,:,t+1)) * J';
end

%%    Post-smoother diagnostics
% --- Smoothed coefficient jumps
dbeta_smooth = nan(nb, T);
for t = 2:T
    dbeta_smooth(:,t) = beta_smooth(:,t) - beta_smooth(:,t-1);
end

% --- Difference between smoothed and filtered beta
smooth_filt_gap = beta_smooth - beta_filt;

end


function [ y_diff, y_diff_KF ] = f_irf_X(j_ex,shock_size,hor,A,A_KF,B,B_KF,flag_plot,fig_no, ...
    flag_ENSO)

[ ~,q   ] = size(B);
[ ~,k,p ] = size(A);

if q>=max(j_ex)                     % run only if the exogenous variable exists
    x_base          = zeros(hor,q); % rows are time
    x_shock         = x_base;

    try flag_ENSO, catch, flag_ENSO = 0; end

    if flag_ENSO
        %  ENSO_t = phi*  ENSO_t-1 + noise
        %  COMM_t = gamma*COMM_t-1 + delta_0*ENSO_t + delta_1*ENSO_t-1 + noise
        phi_0   =  0.79;
        gamma_0 =  0.83;
        % delta_0 =  1.39;
        % delta_1 = -1.72;
        x_ENSO  = zeros(hor,2);
        x_ENSO(1,1) = 1;
        x_ENSO(1,2) = -0.3*x_ENSO(1,1);
        for t=2:hor
            x_ENSO(t,1) = phi_0  *x_ENSO(t-1,1);
            x_ENSO(t,2) = gamma_0*x_ENSO(t-1,2);
        end
        x_shock(:,1:2)  = x_ENSO;
    else
        x_shock(1,j_ex) = x_shock(1,j_ex) + shock_size;
    end
%   y_hist          = Y(end-p+1:end,:); % pxk last p points
%   y_hist          = y_hist*0;
    y_hist          = zeros(max(1,p),k);   % ZERO INITIAL CONDITIONS

    y_base     = forecast_varx(A,    B,    y_hist, x_base,  hor);
    y_shock    = forecast_varx(A,    B,    y_hist, x_shock, hor);
    y_diff     = y_shock    - y_base;      % (hor+1)×k
    if size(A_KF,1)>0
        y_base_KF  = forecast_varx(A_KF, B_KF, y_hist, x_base,  hor);
        y_shock_KF = forecast_varx(A_KF, B_KF, y_hist, x_shock, hor);
        y_diff_KF  = y_shock_KF - y_base_KF;
    end

    if ~isempty(A_KF) && flag_plot
        fig = figure(fig_no); fig.Name = 'IRF';
        subplot(3,2,3), ts  = 1:hor+1;
        plot(ts,y_base,'b-',ts,y_shock,'r--','LineWidth',1.5)
        xlabel('time'), ylabel('irf'), title('Exogenous shock')

        subplot(3,2,4), plot(ts,y_diff,'LineWidth',1.5)
        xlabel('time'), ylabel('D irf'), title('Difference in exogenous shock')

        subplot(3,2,5)
        plot(ts,y_base_KF,'b-',ts,y_shock_KF,'r--','LineWidth',1.5)
        xlabel('time'), ylabel('irf'), title('Exogenous shock (KF)')

        subplot(3,2,6), plot(ts,y_diff_KF,'LineWidth',1.5)
        xlabel('time'), ylabel('D irf')
        title('Difference in exogenous shock (KF)')
        legend('GDP','CPI','FX','EX')
    end
else
    y_diff = []; y_diff_KF = [];
end

end


function [ countries, Y_names, X_names, Y, X, dates ] = load_country_VAR_data( ...
    csv_file, country, idx_XY, flag_plot)

% LOAD_COUNTRY_VAR_DATA
% Reads CSV and returns Y, X matrices for a given country
%
% Inputs
%   csv_file : string, path to CSV file
%   country  : string, e.g. "BRA"
%
% Outputs
%   Y        : T × k endogenous matrix
%   X        : T × q exogenous matrix
%   Y_names  : k×1 cellstr
%   X_names  : q×1 cellstr
%   dates    : table with year, quarter (optional, useful for plotting)

% -----------------------------
% Variable names (AUTHORITATIVE)
% -----------------------------
Y_names = {
    'GDP_YoY'
    'CPI_YoY'
    'FX_YoY'
    'EX_YoY'
    'CPIF_YoY'
};

X_names = {
    'GDP_YoY_star'
    'CPI_YoY_star'
    'FX_YoY_star'
    'EX_YoY_star'
    'ENSO'
    'US_GDP_YoY'
    'CHN_GDP_YoY'
    'COMMODITY_AGR_YoY'
    'PRITHVI_HEAT_STD'
    'PRITHVI_MOISTURE_STD'
    'PRITHVI_MOISTURE_EXTENT'
    'PRITHVI_HEAT_EXTENT'
    'CPI_YoY_annual'
    'FX_YoY_annual'
    'ENSO1'
    'ENSO2'
    'PRITHVI_MOISTURE_STD1'
};
X_names = X_names(idx_XY{1});
Y_names = Y_names(idx_XY{2});
disp(X_names)

% -----------------------------
% Read CSV: country, quarter (MM/dd/uuuu), ENSO, PRITHVI_MOISTURE_STD, PRITHVI_HEAT_STD
%   may have been changed to (uuuu-MM-dd)
% -----------------------------
opts = detectImportOptions(csv_file);
opts = setvartype(opts,'quarter','datetime');
opts = setvaropts(opts,'quarter','InputFormat','uuuu-MM-dd');
T = readtable(csv_file,opts);
% T.ENSO  = T.ENSO > 0.5;                                   % (+) threshold definition
T.ENSO1 = T.ENSO([  1 1:end-1]);                          % ENSO1_t = ENSO_t-1 (1Q lag)
T.ENSO2 = T.ENSO([1 1 1:end-2]);                          % ENSO2_t = ENSO_t-2 (2Q lag)
T.quarter  = datetime(T.quarter,'InputFormat','dd/MM/yyyy');
% idx        = T.quarter.Year > 30;
% T.quarter.Year( idx) = T.quarter.Year( idx) + 1900;
% T.quarter.Year(~idx) = T.quarter.Year(~idx) + 2000;
countries = string(unique(T.country));

% -----------------------------
% Remove quarterly mean
% -----------------------------
flag_remove_quarterly_mean = 0;

% -----------------------------
% Plots
% -----------------------------
if flag_plot || flag_remove_quarterly_mean
    if flag_plot, fig = figure(120); fig.Name = 'climate TH'; end
    for i=1:11
        idx_c     = find(strcmp(T.country,countries(i)));
        % T.GDP_YoY(idx_c) = T.GDP_YoY(idx_c) - mean(T.GDP_YoY(idx_c),'omitmissing');
        % T.CPI_YoY(idx_c) = T.CPI_YoY(idx_c) - mean(T.CPI_YoY(idx_c),'omitmissing');
        % T.FX_YoY( idx_c) = T.FX_YoY( idx_c) - mean(T.FX_YoY( idx_c),'omitmissing');
        % T.EX_YoY( idx_c) = T.EX_YoY( idx_c) - mean(T.EX_YoY( idx_c),'omitmissing');
        moist_c = T.PRITHVI_MOISTURE_STD(idx_c);
        heat_c  = T.PRITHVI_HEAT_STD(idx_c);
        enso_c  = T.ENSO(idx_c);
        for j=1:4
            tbl_moist_m(j) = mean(moist_c(j:4:end),'omitmissing');
            tbl_heat_m( j) = mean( heat_c(j:4:end),'omitmissing');
            T.PRITHVI_MOISTURE_STD( idx_c(j:4:end)) = moist_c(j:4:end) - tbl_moist_m(j);
            T.PRITHVI_HEAT_STD(     idx_c(j:4:end)) =  heat_c(j:4:end) -  tbl_heat_m(j);
        end
        if flag_plot
            subplot(12,2,2*i-1)
%           plot(T.PRITHVI_MOISTURE_EXTENT(idx_c))
            plot(T.PRITHVI_MOISTURE_STD(idx_c)), yline(0)
            ylabel(countries(i)), xticklabels([])
            subplot(12,2,2*i  )
%           plot(T.PRITHVI_HEAT_EXTENT(idx_c))
%           plot(T.PRITHVI_HEAT_STD(idx_c))
            plot(T.quarter(idx_c),T.GDP_YoY(idx_c)), xlim(T.quarter(idx_c([1 end])))
            % xc = xcorr(enso_c(1:end-4),T.PRITHVI_MOISTURE_STD(idx_c(1:end-4)),15);
            % plot(-15:15,xc), yline(0), xline(0)
            % ylabel(countries(i))
            if i<11, xticklabels([]), end
        end
    end

    subplot(12,2,2*i+1), plot(T.quarter(idx_c),enso_c), ylabel('ENSO'), grid on, yline(0)
    % subplot(12,2,2*i+2), plot(T.ENSO(idx_c)), ylabel('ENSO')
end

if isempty(country)
    flag_comm_regress_w = 0;                       % regress commodity on ENSO: world analysis
    flag_comm_regress_c = 1;                       %                            country
    n_countries         = numel(countries);

    if flag_comm_regress_c
        b_EX = nan(3,n_countries);
        p_EX = nan(3,n_countries);

        for i=1:n_countries
            idx  = strcmp(T.country,countries{i});
            comm = T.COMMODITY_AGR_YoY(idx);
            qrts = T.quarter(idx);
            enso = T.ENSO(   idx);
            EX   = T.BAL_YoY( idx);
            Twex = readtable('../data/imf_cache/gvar_macro_CHN');
            idx_wex = ismember(Twex.quarter,qrts);
            wex  = Twex.EX_YoY(idx_wex);               % World exports, use China as a surrogate
            idx  = ~isnan(EX) & ~isnan(wex) & ~isnan(comm);
            qrts = qrts(idx);
            enso = enso(idx);
            comm = comm(idx);
            EX   = EX(  idx);
            wex  = wex( idx);
            enso = (enso - mean(enso))/std(enso);
            comm = (comm - mean(comm))/std(comm);
            EX   = (EX   - mean(EX  ))/std(EX  );
            mdl  = fitlm([enso wex comm],EX);
            b_EX(:,i) = mdl.Coefficients.Estimate(2:end);
            p_EX(:,i) = mdl.Coefficients.pValue(2:end);
        end
    end

    if flag_comm_regress_w
        idx  = strcmp(T.country,countries{1});     % need only 1 country
        qtrs = T.quarter(idx);
        Twex = readtable('../data/imf_cache/gvar_macro_CHN');
        idx_wex = ismember(Twex.quarter,qtrs);
        wex  = Twex.EX_YoY(idx_wex);               % World exports, use China as a surrogate
        comm = T.COMMODITY_AGR_YoY(idx); 
%       comm = T.COMMODITY_YoY(idx); 
%       comm = T.US_GDP_YoY(idx); 
        enso = T.ENSO(idx);  
        i_end= find(isnan(comm),1);
        if ~isempty(i_end)
            comm = comm(1:i_end-1);
            enso = enso(1:i_end-1);
            wex  = wex( 1:i_end-1);
            qtrs = qtrs(1:i_end-1);
        end
        i_beg= find(~isnan(wex),1);
        comm = comm(i_beg:end);
        enso = enso(i_beg:end);
        wex  = wex( i_beg:end);
        qtrs = qtrs(i_beg:end);
        enso = (enso - mean(enso))/std(enso);
        comm = (comm - mean(comm))/std(comm);
        wex  = (wex  - mean(wex ))/std(wex );
        enso_p = max(0,enso);
        enso_n = min(0,enso);
        mdlwex = fitlm(wex,                              comm       );
        mdl0   = fitlm([enso_p          enso_n         ],comm       );
        mdl1   = fitlm([enso_p(1:end-1) enso_n(1:end-1)],comm(2:end));
        mdla0  = fitlm( enso,                            comm       );
        mdla1  = fitlm( enso(  1:end-1),                 comm(2:end));
        % beta ( ENSO -> COMMODITY_YoY ) = -0.286, p < 0.0001
        %                World_EX_YoY      -0.278, p = 0.0026
        % no significant relationship with US GDP (except for negative enso)

        y   = comm; ylbl = 'Commodities';
        fig = figure(99); fig.Name = 'ENSO->y';
        subplot(2,2,1)
        plot(enso,y,'bo'), xlabel('ENSO'), ylabel(ylbl), grid on, xline(0), yline(0)

        subplot(2,2,3:4)
        plot(qtrs,[enso y]), xlabel('quarters'), ylabel('anomaly'), grid on
        legend('ENSO',ylbl)

        n_qrts    = 30;                                    % number of lags
        [ xy,tc ] = xcorr(enso,y,   n_qrts,"normalized");
        [ xx    ] = xcorr(enso,enso,n_qrts,"normalized");
        [ yy    ] = xcorr(y,   y,   n_qrts,"normalized");

        subplot(2,2,2), plot(tc/4,[xy xx yy]), xlabel('years'), ylabel('correlation')
        title(['Cross-correlation between ENSO and ' ylbl]), grid on
        xline(0), yline(0)
        % ENSO -> Commodities with time lag shows correlation for negative time lags

        fig = figure(98); fig.Name = 'scatterplot';
        p = plot3(wex,comm,enso,'bo'); xlabel('World EX YoY'), ylabel('Ag prices YoY')
        sequence = 1:length(enso); xline(0), yline(0)
        % 3. Create a new DataTip row
        % The first argument is the Label, the second is the data to display
        row = dataTipTextRow('Point No:', sequence);
        p.DataTipTemplate.DataTipRows(end+1) = row;

        keyboard
    end
    dates = []; X = []; Y = [];
    return
end
% -----------------------------
% Filter country
% -----------------------------
T = T(strcmp(T.country, country), :);

if isempty(T)
    error('No data found for country %s', country);
end

% -----------------------------
% Sort by time
% -----------------------------
% quarter is a date string like "1/1/97"
if ismember('quarter', T.Properties.VariableNames)
    T.quarter = datetime(T.quarter, 'InputFormat', 'M/d/yy');
    T = sortrows(T, {'year', 'quarter'});
else
    T = sortrows(T, 'year');
end

% -----------------------------
% Shift in time
% -----------------------------
T.PRITHVI_MOISTURE_STD1 = T.PRITHVI_MOISTURE_STD([2:end end]); % t=1 in STD1 => t=2 in STD

% -----------------------------
% Keep only needed columns
% -----------------------------
keep_vars = [{'year', 'quarter'}, Y_names', X_names'];
keep_vars = keep_vars(ismember(keep_vars, T.Properties.VariableNames));
T = T(:, keep_vars);

% -----------------------------
% Drop rows with missing data
% -----------------------------
T = rmmissing(T, 'DataVariables', [Y_names; X_names]);

% -----------------------------
% Build Y and X matrices
% -----------------------------
Y = T{:, Y_names};   % T × k
X = T{:, X_names};   % T × q

% -----------------------------
% Optional date output
% -----------------------------
if ismember('quarter', T.Properties.VariableNames)
    dates = T(:, {'year', 'quarter'});
else
    dates = T(:, {'year'});
end

end


function L = forecast_loss(theta,x,y,Beta,P0,Sigma_u,p,t1,tc,H,DY,flag_VAR,shrink,eps_scale)

lambda = theta(1);
R_mat  = theta(2);

[ ~,k ] = size(y); [ ~,q ] = size(x); n = k*p + q;

% run KF with these hyperparameters
if flag_VAR==1
    [ ~,~,beta_filt] = f_KF(x,y,Beta,P0,Sigma_u,R_mat,lambda,p);
else
    beta_filt        = [];
end

% rolling 4-step forecast
yhat = rolling_forecast(t1,p,H,y,x,DY,flag_VAR,shrink,eps_scale,R_mat,lambda,beta_filt);

% loss (example: MSE at h=2)
err = y(tc:end,:) - yhat(tc:end,:,H);
L   = mean(err(:).^2);

end


function [ yhat,Yhat,B_ENSO_time ] = rolling_forecast(t1,p,H,y,x,DY,flag_VAR, ...
    shrink,eps_scale,R_mat,lambda,beta_filt)

[ T,k ]  = size(y); [ ~,q ] = size(x);

Yhat = nan(T, k, H); yhat = Yhat; Beta = [];
% enso_idx    = 1:3;
enso_idx    = 1;
% fprintf('----- enso_idx = %i -----\n',enso_idx)
B_ENSO_time = nan(T, k, numel(enso_idx));

for t = t1:(T-H)        % start with data from 1:t1 and end with data from 1:T-H
    y_est = y(1:t,:);   % collect data from 1:t
    x_est = x(1:t,:);
    if flag_VAR~=1
        % --- fit VAR
        res = estimate_VARX(y_est, x_est, p);
    end
    if flag_VAR==0      % VARX
        A       = res.A;
        B       = res.B;
    elseif flag_VAR==1  % use KF beta coefficients at time t
        [ A,B ] = extract_AB(beta_filt(:,t), p, k, q);
    elseif flag_VAR>=2  % use Beta from VAR which uses data up to time t
        Sigma_u = res.Sigma_u; ZZ = res.ZZ;
        if flag_VAR==2 || flag_VAR==3 && isempty(Beta) % flag_VAR=3 => use Beta from t=t1
            Beta = res.Beta;
        end
        P0      = f_P0(ZZ,Sigma_u,p,shrink,eps_scale);
        [ A,B ] = f_KF(x_est,y_est,Beta,P0,Sigma_u,R_mat,lambda,p);
    end

    % store lag-specific ENSO coefficients
    B_ENSO_time(t,:,:) = B(:, enso_idx);     % k × (# ENSO lags)

    % --- last p observations (standardized)
    y_buff = y_est(t-p+1:t,:); % previous p observations up to time t
%   x_next = x(    t+1,:);     % next exogenous
    x_last = x_est(t,:);       % last exogenous at time t
    % --- H-step forecast (standardized)
    for h=1:H
        yhat_h = zeros(1,k);
        for lag = 1:p
            yhat_h = yhat_h + ...
                (A(:,:,lag) * y_buff(end-lag+1,:).')';
        end
        yhat_h = yhat_h + (B * x_last.').';
        % latest data is at t = T-H, t+h goes up to T
        yhat(t+h,:,h) = yhat_h;
        Yhat(t+h,:,h) = (DY * yhat_h')';            % unstandardize
        y_buff        = [y_buff(2:end,:) ; yhat_h]; % add last estimate to Y_buff
    end
end

end


function [ Y,X,A,B,u ] = generate_VARX(k,q,p,T)

Y        = zeros(T,k);
X        = randn(T,q);
A        = zeros(k,k,p);
B        = zeros(k,q);
u        = randn(T,k);    % innovation

A(:,:,1) = eye(k)*0.7;    % begin with scaled identity for A1
B(:,1)   = -1.0;
A(1,2,1) = -0.2;
A(2,1,1) =  0.1;
if p>1
    A(1,2,2:p) = 0.2
end

u        = u*0.1;  % scaled innovation

for t=p+1:T                                    % Y at next time t
    Y(t,:) = X(t-1,:)*B' + u(t-1,:);           %   depends on X at time t-1
    for j=1:p
        % need to take transpose of X, Y
        Y(t,:) = Y(t,:) + Y(t-j,:)*A(:,:,j)';  %   and Y at times t-1 ... t-p
    end
end

flag_plot = 0;
if flag_plot
    fig = figure(99); fig.Name = 'generate VARX data';
    subplot(3,1,1), plot(1:T,Y), title('Y')
    subplot(3,1,2), plot(1:T,X), title('X')
    subplot(3,1,3), plot(1:T,u), title('innovation')

    fprintf('\n abs eig A1:\n')
    disp(abs(eig(A(:,:,1))))
end

end


function res = estimate_VARX(Y, X, p)

% ESTIMATE_VARX  Estimate VARX(p) by OLS (no constant), with standardization
%
% Inputs
%   Y_raw : T × k matrix of endogenous variables
%   X_raw : T × q matrix of exogenous variables
%   p     : lag order
%
% Output (struct)
%   res.A        : k × k × p   lag coefficient matrices
%   res.B        : k × q       exogenous coefficient matrix
%   res.Beta     : (p*k + q) × k stacked coefficient matrix
%   res.Sigma_u  : k × k       innovation covariance
%   res.stdY     : 1 × k       std devs of Y
%   res.stdX     : 1 × q       std devs of X
%   res.DY       : k × k       diag(stdY)
%   res.DX       : q × q       diag(stdX)
%   res.Z        : (T-p) × (p*k + q) regressor matrix
%   res.Y_reg    : (T-p) × k   regression target
%
% Notes
%   • No constant
%   • Standardization matches your Python + MATLAB pipeline
%   • Beta layout matches statsmodels VARX ordering:
%       [Y_{t-1}, ..., Y_{t-p}, X_t]

% -----------------------------
% Dimensions
% -----------------------------
[T, k] = size(Y);
q      = size(X, 2);
n      = k*p + q;         % number of input variables Y_t-1, ..., Y_t-p, X

% -----------------------------
% Build lagged Y matrix
% -----------------------------
Ylag = [];
for i = 1:p
    Ylag = [Ylag, Y(p+1-i:T-i, :)]; % [ Y(p:T-1,:) Y(p-1:T-2,:) ... ] (T-p, k*p)
end
t0     = max(p,1);           % if p>=1, the 
if q>0
    X_lag  = X(t0:T-1, :);       % corresponds to Ylag_1
else
    X_lag  = [];
end

% -----------------------------
% Align regression data
% -----------------------------
Y_next = Y(t0+1:T, :);       % (T-p, k)

% Final regressor matrix (NO constant)
% [ Y(p:T-1,:) Y(p-1:T-2,:) ... Y(1:T-p,:) X(p:T-1,:) ]
Z = [Ylag, X_lag];           % (T-p, k*p+q) = (T-p, n)

% -----------------------------
% OLS estimation
% -----------------------------
ZZ   = Z' * Z;               % (n, n)

flag_ridge = 0;

if flag_ridge
    if p>1, keyboard, end
    lambda_ridge = 100;

    A = zeros(k,k);
    B = zeros(k,q);

    for i = 1:k
        yi = Y_next(:,i);    % dependent variable (T-p,1)

        % Ridge target: a_i ≈ e_i
        target    = zeros(k+q,1);
        target(i) = 1;      % diagonal element of A

        % Ridge matrix
        R      = lambda_ridge * eye(k+q);

        % Ridge-regularized OLS
        beta_i = (Z' * Z + R) \ (Z' * yi + lambda_ridge * target);

        A(i,:) = beta_i(1:k)';
        B(i,:) = beta_i(k+1:end)';
    end
    Beta = [ A' ; B'];            % (p*k+q,k)
else
    Beta = ZZ \ (Z' * Y_next);    % (p*k+q,k) = (n,k)
end

U = Y_next - Z * Beta;

% Innovation covariance (df-adjusted)
Sigma_u = (U' * U) / (size(U,1) - size(Z,2)); % divide by (T-p - n) so T > p+n

%% EXTRACT A AND B
%  Beta   = [          A' ;            B' ]
%  Z*Beta = [ Z(1:k*p)*A' ; Z(k*p+1:n)*B' ]
Ap = Beta(1:k*p,  :);            % A' (k*p,k)
Bp = Beta(k*p+1:n,:);            % B' (q,  k)

A  = reshape(Ap',k,k,p);
B  = Bp';                        % k × q

% -----------------------------
% Package results
% -----------------------------
res.A     = A; res.B = B; res.Z = Z; res.ZZ = ZZ; res.p = p; res.Sigma_u = Sigma_u;
res.Y_reg = Y_next;
res.Beta  = Beta;              % Z*Beta or Beta'*Z(:)

end


function y_hat = forecast_varx(A, B, y_hist, x_future, H)

% Horizon H refers to the available future data
% Forecasts begin at p+1 and end at H+1
% y_hist is used to generate the prediction at p+1

% y_hist:   (p,  k) with last row = y_t (most recent)
% y_hat:    (H+1,k) including projection at time H+1
% x_future: (H,  q) for t=1...H

[k,~,p]    = size(A);
[T,  q]    = size(x_future);

y_hat      = zeros(H+1, k);

% y_hat(1,:) = y_hist(end,:);                     % no need to use last value of y_hist
% y_hat      = [ y_hist(end,:) ]
%              [ 0 ...       0 ]

% maintain a rolling buffer of past p y's (oldest -> newest)
% y_buf    = [ y_hist(1,:)   ]
%                ...
%            [ y_hist(p,:)   ]

y_buf      = y_hist;                              % (p,k) always with rows added & removed

%% PREDICTIONS USING ROLLING p ROWS OF Y (y_buf) TO PREDICT AT TIME h+1
%  Initial y_hat(1:p,:) = 0 (incomplete data at previous p times)
for h_data = p:H                                  % if H = T, last prediction at T+1
    h_projection = h_data + 1;
    y_next = zeros(1,k);                          % initialize next predicted y
    for lag = 1:p
        % lag=1 => newest Ybuf => last row=end
        y_next   = y_next + y_buf(end-lag+1,:) * A(:,:,lag)';
    end
    if q>0
        y_next   = y_next + x_future(h_data,:) * B';
    end
    y_hat(h_projection,:) = y_next;

    % update buffer
    y_buf       = [y_buf(2:p,:); y_next];
end

end


function [ irf,IRF ] = f_irf_Y(H,A,DY)

% Impulse response for each component of Y
% H   = horizon
% DY  = diagonal matrix of std Y
% irf = impulse response in Y normalized by DY
% IRF =                  in Y original coordinates

[ k,~,p ]  = size(A);

irf        = zeros(H+1,k,k); IRF = irf;
irf(1,:,:) = eye(k);                     % impulse response at time 0 for each Y component

for h=1:H                                % h=1 => time 0
    Psi = zeros(k);
    for lag=1:min(h,p)                   % sum over all lags, not going before time 0
        Psi = Psi + A(:,:,lag) * squeeze(irf(h-lag+1,:,:));
    end
    irf(h+1,:,:) = Psi;                  % save in next time step
end
for h=1:H+1
    IRF(h,:,:) = DY * squeeze(irf(h,:,:)) * inv(DY);
end

end


function [ A,B,beta_filt,a_filt ] = f_KF(x,y,Beta0,P0,Sigma_u,R_mat,lambda,p,...
    V,n_SVD,beta_mean0)

[ T,k ]   = size(y); [ ~,q ] = size(x);

try
    assert(n_SVD>0);                  % error if n_SVD = 0 or was not passed
    flag_SVD  = true;
catch
    flag_SVD  = false;
end

if flag_SVD
    nb        = n_SVD;
    V         = V(:,1:n_SVD);         % take subspace
    beta_pred = V'*Beta0(:);          % use V subspace and convert back to whole space later
    P_pred    = V'*P0*V;
    beta_mean = V'*beta_mean0(:);     % V subspace (beta_mean), whole space (beta_mean0)
else
    n         = k*p + q;
    nb        = n*k;
    beta_pred = Beta0(:);
    P_pred    = P0;    % your diagonal prior
    beta_mean = 0;
end

beta_filt = zeros(nb, T);
P_filt    = zeros(nb, nb, T);

% V(nb,n_SVD), V'*V = I(n_SVD), beta_pred = V*a_pred, a_pred = V'*beta_pred
% P_beta = cov(beta) = E[beta*beta'] (assuming zero means)
% P_a    = cov(a)    = E[   a*a'   ] = V'*E[beta*beta']*V = V'*P_beta*V
% Zt     *beta_filt  = (k,   nb)*(nb,   1)
% Zt*V*V'*beta_filt  = (k,n_SVD)*(n_SVD,1)

I = eye(nb);

% Measurement noise covariance
if isscalar(R_mat)
    R = R_mat * Sigma_u;   % k × k
else
    R = R_mat;             % already k × k
end

% Joseph-stabilized covariance update
t0 = max(p,1);                         % allow for p=0

for t = t0+1:T                         % start with yt at time t0+1
    % --- Prediction, use P*(1/lambda - 1) in place of Q
    P_pred = P_pred / lambda;

    % --- Measurement
    % ZVt = Zt*V
    Zt    = build_Zt(y, x, t, p);      % (k,   nb_orig), time t-1 ... t-p
    if flag_SVD
        Zt = Zt * V;                   % (k,n_SVD)
    end
    H = Zt;

    % --- Innovation
    % V_stacked: beta_pred = V*a_pred, but need to include Beta_mean
    %            ZVt = Zt*V
    yt    = y(t,:)';                   % (k, 1), time t
    v     = yt - H * (beta_pred + beta_mean);

    % --- Innovation covariance (WITH R_mat)
    S = H * P_pred * H' + R;           % (k,k) = (k,nb)*(nb,nb)*(nb,k) + (k,k)

    % Kalman gain
%   K = P_pred * H' / S;
    K = P_pred * H' * pinv(S);

    % --- Update
    beta_filt(:,t) = beta_pred + K * v;
    KH             = K * H;
    P_filt(:,:,t)  = ...
        (I - KH) * P_pred * (I - KH)' + K * R * K';

    % Prepare next iteration
    beta_pred = beta_filt(:,t);
    P_pred    = P_filt(:,:,t);
end

if flag_SVD
    a_filt    = beta_filt;
    beta_filt = V *a_filt;              % put in V subspace, beta = V*V'*beta = V*a
    beta_filt = beta_filt + beta_mean0(:);
else
    a_filt    = [];
end

% A,B
% beta_filt = V*a_filt
[ A,B ] = extract_AB(beta_filt(:,T), p, k, q);

end


%% Take out SVD in f_KF()
function beta_filt = f_KF_basic(x,y,Beta0,P0,Sigma_u,R_scale,lambda,p)

%  TVP-VARX Time series equations (for the special case of lag p = 1)
%  y_t+1   = A*y_t + B*x_t    + u_t
%          = beta*[y_t ; x_t] + u_t
%
%  Kalman Filter parameters
%  Beta0   = initial value for beta
%  P0      = diagonal matrix for the prior variance of beta
%  Sigma_u = variance of u_t
%  R_scale = scalar multiple used with Sigma_u for the variance of the innovation R
%  lambda  = forgetting factor used to increase the variance of beta P at each time step
%  p       = maximum lag of the endogenous variables y

[ T,k ]   = size(y); [ ~,q ] = size(x);

n         = k*p + q;
nb        = n*k;
beta_pred = Beta0(:);
P_pred    = P0;                        % your diagonal prior

beta_filt = zeros(nb, T);
P_filt    = zeros(nb, nb, T);
I         = eye(nb);

% Measurement noise covariance
R = R_scale * Sigma_u;                 % k × k

% Joseph-stabilized covariance update

for t = 2:T                            % start with yt at time 2
    P_pred = P_pred / lambda;

    % --- Measurement
    H     = build_Zt(y, x, t, p);      % (k,   nb_orig), time t-1 ... t-p

    % --- Innovation
    yt    = y(t,:)';                   % (k, 1), time t
    v     = yt - H * beta_pred;

    % --- Innovation covariance
    S = H * P_pred * H' + R;           % (k,k) = (k,nb)*(nb,nb)*(nb,k) + (k,k)

    % --- Kalman gain
    K = P_pred * H' * pinv(S);

    % --- Update
    beta_filt(:,t) = beta_pred + K * v;
    KH             = K * H;
    P_filt(:,:,t)  = (I - KH) * P_pred * (I - KH)' + K * R * K';

    % --- Prepare next iteration
    beta_pred = beta_filt(:,t);
    P_pred    = P_filt( :,:,t);
end

end


%% Kalman smoother
function [beta_filt,beta_smooth,P_filt,P_smooth,v_all,S_all,innov_score, ...
          dbeta_smooth,smooth_filt_gap] = ...
    f_KF_smoother(x,y,Beta0,P0,Sigma_u,R_scale,lambda,p)

%  TVP-VARX Time series equations (for the special case of lag p = 1)
%  y_t+1   = A*y_t + B*x_t    + u_t
%          = beta*[y_t ; x_t] + u_t
%
%  Kalman Filter parameters
%  Beta0   = initial value for beta
%  P0      = diagonal matrix for the prior variance of beta
%  Sigma_u = variance of u_t
%  R_scale = scalar multiple used with Sigma_u for the variance of the innovation R
%  lambda  = forgetting factor used to increase the variance of beta P at each time step
%  p       = maximum lag of the endogenous variables y

[ T,k ]   = size(y);
[ ~,q ]   = size(x);

n         = k*p + q;
nb        = n*k;

beta_pred = Beta0(:);
P_pred    = P0;                        % your diagonal prior

beta_filt = zeros(nb, T);
P_filt    = zeros(nb, nb, T);
I         = eye(nb);

% --- store predicted quantities for smoother
beta_pred_all = zeros(nb, T);
P_pred_all    = zeros(nb, nb, T);

% --- store innovation diagnostics
v_all        = zeros(k, T);
S_all        = zeros(k, k, T);
innov_score  = nan(1, T);              % v' * inv(S) * v

% --- Measurement noise covariance
R = R_scale * Sigma_u;                 % k × k

% --- Initialize time 1 with the prior
beta_filt(    :,1) = Beta0(:);
beta_pred_all(:,1) = Beta0(:);
P_filt(    :,:,1)  = P0;
P_pred_all(:,:,1)  = P0;

%%    Kalman filter
% --- Joseph-stabilized covariance update
for t = 2:T                            % start with yt at time 2
    % --- Prediction step
    P_pred = P_pred / lambda;

    % Since state transition is identity: beta_pred stays equal to previous filtered beta
    % (already stored in beta_pred from previous iteration)

    % --- Store predicted quantities for smoothing
    beta_pred_all(:,t) = beta_pred;    % beta_pred_all(:,t) = beta_filt(:,t-1)
    P_pred_all( :,:,t) = P_pred;       % P_pred_all(   :,t) = P_filt(   :,t-1)

    % --- Measurement
    H     = build_Zt(y, x, t, p);      % (k, nb), time t-1 ... t-p

    % --- Innovation
    yt    = y(t,:)';                   % (k,1), time t
    v     = yt - H * beta_pred;

    % --- Innovation covariance
    S = H * P_pred * H' + R;           % (k,k)

    % --- Store innovation diagnostics
    v_all(:,t)     = v;
    S_all(:,:,t)   = S;
    innov_score(t) = real(v' * (S \ v));

    % --- Kalman gain
    K = P_pred * H' / S;

    % --- Update
    beta_filt(:,t) = beta_pred + K * v;
    KH             = K * H;
    P_filt(:,:,t)  = (I - KH) * P_pred * (I - KH)' + K * R * K';

    % --- Prepare next iteration
    beta_pred = beta_filt(:,t);
    P_pred    = P_filt( :,:,t);
end

%%    RTS smoother
beta_smooth = zeros(nb, T);
P_smooth    = zeros(nb, nb, T);

% --- Start backward pass from final filtered estimate
beta_smooth(:,T) = beta_filt(:,T);
P_smooth( :,:,T) = P_filt( :,:,T);

for t = T-1:-1:1
    % Since state transition matrix F = I, smoother gain is:
    % J_t = P_filt(:,:,t) * inv(P_pred_all(:,:,t+1))
    J = P_filt(:,:,t) / P_pred_all(:,:,t+1);

    % --- Smoothed state
    beta_smooth(:,t) = beta_filt(:,t) + J * (beta_smooth(:,t+1) - beta_pred_all(:,t+1));

    % --- Smoothed covariance
    P_smooth(:,:,t) = P_filt(:,:,t)   + J * (P_smooth(:,:,t+1) - P_pred_all(:,:,t+1)) * J';
end

%%    Post-smoother diagnostics
% --- Smoothed coefficient jumps: useful for structural break detection
dbeta_smooth = nan(nb, T);
for t = 2:T
    dbeta_smooth(:,t) = beta_smooth(:,t) - beta_smooth(:,t-1);
end

% --- Difference between smoothed and filtered beta:
%     large values indicate that future information materially revised the state
smooth_filt_gap = beta_smooth - beta_filt;

end


function P0 = f_P0(ZZ,Sigma_u,p,shrink,eps_scale)

n      = size(ZZ,1); k = size(Sigma_u,1); nb = k*n; q = n - p*k;

ZZ_inv = inv(ZZ);               % (n,n)
cov_sm = kron(Sigma_u, ZZ_inv); % size: (nb × nb)
cov_sm = cov_sm * n;            % DEBUG: make it similar in magnitude to Python
blocks = {};
% --- lag blocks
for lag = 1:p
    block = zeros(k);
    for i = 1:k          % equation
        for j = 1:k      % variable
            idx = (i-1)*n + (lag-1)*k + j;
            block(j,i) = cov_sm(idx, idx);
        end
    end
    blocks{end+1} = block;
end
% --- exogenous block
if q > 0
    B_block = zeros(q, k);
    for i = 1:k
        for j = 1:q
            idx = (i-1)*n + p*k + j;
            B_block(j,i) = cov_sm(idx, idx);
        end
    end
    blocks{end+1} = B_block;
end
% --- stack into Kalman ordering
P0_diag = [];
for b = 1:length(blocks)
    P0_diag = [P0_diag; blocks{b}(:)];   % column-major = order="F"
end

P0 = diag(P0_diag);

P0 = shrink * P0;
P0 = P0 + eps_scale * eye(size(P0,1));

end


function Zt = build_Zt(Y, X, t, p)

% Zt * beta_filt = (k,nb) * (nb,1)
%
% beta' = [ A11 A12 ... A1k B11 ... B1q   A21 ... B2q   ...  Ak1 ... Bkq     ]
%       = [ A( 1,:)         B( 1,:)       A(2,:)  B( 2,:)    A( k,:) B( k,:) ]
%       = [ A'(:,1)         B'(:,1)       A'(:,2) B'(:,2)    A'(:,k) B'(:,k) ]
%       =   Beta(:)'
% Beta  = [ A' ; B' ]
% Zt    = [ Y1  Y2 ...  Yk  X1 ...  Xq
%                                         Y1 ...  Xq
%                                                       ...
%                                                            Y1 ...  Xq      ]

k = size(Y,2);
q = size(X,2);

z_row = [];
for lag = 1:p
    z_row = [z_row, Y(t-lag,:)];
end
z_row = [z_row, X(t-1,:)];   % 1 × n

Zt = kron(eye(k), z_row);    % k × (k*n)

end


function [ A,B ] = extract_AB(beta_t, p, k, q)

% Beta   = [ A    B  ]' (n,k) for p=1
%        = [ A' ]
%          [ B' ]
% 
%        = [ A11 A12 ... A1k B11 ... B1q
%            A21 A22 ...             B2q
%                    ...
%            Ak1 Ak2 ...             Bkq ]'
% 
%        = [ A11 A21 ... Ak1
%            A12 A22 ... Ak2
%                    ...
%            A1k A2k ... Akk
%            B11 B21 ... Bk1
%                    ...
%            B1q B2q ... Bkq ]
%
% beta_t = [ A11 A12 ... A1k B11 ... B1q   A21 ... B2q   ...  Ak1 ... Bkq]

n        = p*k + q;
Beta_t   = reshape(beta_t, n, k);   % rows are equations
B        = Beta_t(p*k+1:end,:)';    % k × q
A        = zeros(k,k,p);

for lag = 1:p
    rows       = (lag-1)*k + (1:k);
    A(:,:,lag) = Beta_t(rows,:)';
end

end
