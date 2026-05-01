function debug_MATLAB_2026_03_24()

%% STRUCTURAL BREAKS
%  Flag: flag.fig_pdf     (below, not necessary, takes time)
%        flag_plot_detail (in f_results(), only if needed overwrites Figure 99)
%        flag_write...

%% HEIRARCHICAL BAYESIAN APPROACH
%  Flags: 0 0 1 0 0 0 0 1 1 (for ENSO) 1 (clustering)
%  Regress commodity on ENSO in load_country_VAR_data() by setting flag_comm_regress=1
%    Result: beta ( ENSO -> COMMODITY_YoY ) = -0.286, p < 0.0001

%% ASSUMPTIONS
%  B non-zero ENSO effects:              ENSO -> CPI (up), EX (down)
%  A non-zero off-diagonal coefficients: GDP  -> EX
%                                        FX   -> CPI
%% CLIMATE EFFECTS
%  Egypt:     95% irrigation so precipitation has little effect except if it affects the Nile

%% SHOCKS AND SHIFTS
%  A. Temporary exogenous shock (omitted variable)
%  	 	large innovation
% 	 	temporary coefficient movement
% 	 	coefficients revert quickly
%  B. Structural change
% 	 	persistent shift in coefficients
% 	 	new regime remains

%% FLAGS
flag.SVD_A_only    = 0; % 0 use SVD for A and retain all coefficients in B
flag.simulate_data = 0; % 0
flag.macro         = 1; % 0 use macro assumptions (not too sensitive)
flag.KF_in_SVD     = 0; % 1 perform KF in SVD space (uses nSVD1)
flag.KF_projected  = 0; % 0 project KF in SVD coordinates
flag.SVD           = 0; % 1
flag.macro_no_X    = 0 & flag.macro; %  1   use macro assumptions in the no_X analysis

csv_file = '../data/gvar_panel_streamlit.csv'; % data for GVAR

%% INITIALIZE PARAMETERS
% IRF: j_ex = 1: ENSO, j_ex = 2: agr commodities (see idx_X)
j_ex               = 1;              %  2
shock_size         = 1;              %
flag.fig_pdf       = 1;

p        = 1; % Order of autoregression

% 5: ENSO, 15: ENSO1, 8: agricultural commodities, 6: US GDP
idx_X   = [ 15  8 6   ]; % ENSO1, agr commodities (ENSO1 better than ENSO), US_GDP
idx_Y   = [ 1 2 4 3   ]; % GDP CPI EX FX *** CPI SEEMS MORE REASONABLE THAN CPIF ***
idx_XY  = {idx_X,idx_Y};

%% KF PARAMETERS
%  P0*shrink uncertainty in A,B
%  lambda forgetting factor
%  R variance of innovation of y
%  eps_scale = noise term
lambda      =  1; shrink = 1; eps_scale = 0.001; % is more stable than VAR A,B
R_mat       = 10;           % R matrix
hor         = 12;           % horizon for IRFs (e.g. 12 quarters)

% if T_covid_end = 0, then use all data
T_covid_end     = 20*0;     % save quarters up to T_covid: X = X(1:end-T_covid,:);

flag.plot_irf   = 1;
fig_no_irf      = 101;      % figure for plotting data
flag.plot       = 0;

%  Get countries, variables names, parameters (initialize using input countries = [])
%    k  = number of endogenous variables
%    q  = number of exogenous variables
%    n  = p*k + q   regressors per equation
%    nb = k * n     total state dimension

%% INITIALIZE countries,k,q,n,nb
[ countries,Y_names,X_names,k,q,n,nb ] = ...
    f_results(csv_file,[],p,idx_XY,j_ex,shock_size,hor,shrink,eps_scale,R_mat,lambda,T_covid_end,flag.plot); 

disp(Y_names)
disp(X_names)

% to do all countries, comment out the next two lines
countries_analyze = ["BRA","CHL","COL","IND","IDN","PHL"]; % subset of countries to analyze
countries         = countries(ismember(countries,countries_analyze));
n_countries       = numel(countries);                      % number of countries

%% MULTIPLE COUNTRY ANALYIS: KF and VARX
Beta0      = [ eye(k) ; zeros(k*(p-1),k) ; zeros(q,k) ];   % Y and X (n,k)
Beta0      = Beta0 * 0.25;          % scale identity
A_KFs      = nan(k,k,max(1,p),  0); % allow zeros if p=0 [REPLACE n_countries WITH 0]
A_KFs_no_X = A_KFs;
B_KFs      = nan(k,q,           0);
B_KFs_no_X = B_KFs;
y_diff_KFs = nan(hor+1,k,0);
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
        if ~isnan(Betas(1,1,i))
            i_good = 1;                       % good data found
        end
    catch ME                                  % if no outputs from f_results(), continue
        if ~strcmp(ME.identifier,'MATLAB:unassignedOutputs'), keyboard, end
    end
    
    if i_good
        i = i+1;                              % increment index and continue
    else
        i_not_enough_data = [i_not_enough_data i];
        countries   = countries(setdiff(1:n_countries,i)); % take out country i from countries
        n_countries = n_countries - 1;
        i_good      = 0;
    end
end

if ~isempty(i_not_enough_data)
    fprintf('Countries with not enough data:\n'), fprintf('%s, ',countries0(i_not_enough_data))
    fprintf('\n')
end

hor_plot = 1:7;
fig      = figure(138); fig.Name = 'IRFs'; clf
tiledlayout('flow','TileSpacing','tight','Padding','tight');

y_diffs = nan(n_countries,k);

%% CALCULATE AND PLOT IRFs
for i=1:n_countries
    y_diff = f_irf_X(j_ex,shock_size,hor,A_KFs( :,:,1,i),[],B_KFs( :,:,i),[],[],[],0);
    nexttile
    plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
    y_diffs(i,:) = y_diff(2,:);
end

for i=1:n_countries
    fprintf('%.3f,%.3f,%.3f,%.3f,%s\n',y_diffs(i,:),countries{i})
end

flag_clustering_and_other_analysis = 0;

if flag_clustering_and_other_analysis
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
    flag.clustering    = 1;            %

    jBkeep  = [   2 3     ]; % 1:4 indices of B (ENSO        -> A) to keep
    jBkeep  = [ jBkeep 6:8]; % 5:8              (commodities -> A)
    jBkeep  = [ jBkeep 9:12];% US GDP

    % analysis without X
    for i=1:n_countries
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
    end

    %% Betas FROM ALL COUNTRIES
    T_min      = min(Ts);                % start w/83 countries, reduce to 71 (not enough data)
    Betas_mean = mean(Betas,3);          % average betas from standard VAR

    %  Plot Betas from KF, VARX
    clims      = [-0.5 0.8];
    Beta_sim   = [];                     % need this to prevent problems later

    %  Use final A,B from KF
    %  Stack these rows from final KF time: [ A(:,1) ; ... ; A(:,k) ; B(:,1) ; ... ; B(:,q) ]'
    beta_stacked = nan(n_countries,nb); % n = p*k + q, nb = n*k
    for i=1:n_countries
        beta_stacked(i,:) = beta_filts{i}(:,round(Ts(i)*0.8));
    end

    %% USE SELECT COUNTRIES
    for i=1:n_countries, fprintf('%2i %s\n',i,countries(i)), end

    beta_mean_select= 1:n_countries;
    % beta_mean_select  = [2 4 8 9 15 16];
    % beta_mean_select  = [3 4 7 8 11:17];
    beta_stacked_mean = mean(beta_stacked(beta_mean_select,:));
    beta_stacked      = beta_stacked - beta_stacked_mean;

    %% KF IN SVD SPACE
    n_SVD1      = k+2+numel(jBkeep)+2;              % diag, 2 in A, * in B, 2 more in A
    sz          = [n,k];                            % size of Beta
    V_stackedA  = zeros(nb,n_SVD1);                 % corresponds to [ A' ; B' ](:)
    V_stackedB  = zeros(nb,n_SVD1);
    for j=1:k
        V_stackedA(sub2ind(sz,j,j), j) = 1;         % 1, n+2, 2*n+3, ...
    end
    %  rows of Beta: GDP, CPI, FX, EX, ENSO (input)
    %  cols of Beta: GDP, CPI, FX, EX       (output)
    %   V_stackedA(sub2ind(sz,1,3), k + 1) = 1;         % GDP (row 1) -> EX (col 3)
    %   V_stackedA(sub2ind(sz,4,2), k + 2) = 1;         % FX  (row 4) -> CPI(col 2)
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
    if flag_mean_select
        n_iter = 2;
    end
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
        end

        % 0.10 => 27 out of 77 countries
        % 0.25    57
        lim_keep            = 0.25;                                  % decay upper limit at hor
        [ idx_keep,n_keep ] = analysis_A(A_KFVs,hor,lim_keep);

        %% SECOND ITERATION, USE DIFFERENT beta_stacked_mean USING beta_mean_select COUNTRIES
        if n_iter>1 && i_iter==1
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

            % kmedoids handles noisy rows much better than kmeans
            [C1_cl, M_cl]   = kmeans(U1_cl, n_cl, 'Distance', 'cityblock');

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

    %% PLOT BETAS [DISABLED]
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
    hor_plot = 1:7;

    n_col_plot = n_countries;

    if n_countries>20
        flag_plot_all = 0;                         % don't squeeze too many IRFs into a figure
    else
        flag_plot_all = 1;
    end

    fig      = figure(137); fig.Name = 'IRFs with KF using SVD'; clf
    if flag_plot_all
        tiles = tiledlayout(4,n_col_plot,'TileSpacing','tight','Padding','tight');
    else
        tiledlayout('flow','TileSpacing','tight','Padding','tight');
    end

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

            %% IRF BASED ON BETAS FROM VARX AND KF USING
            %%   ENDOGENOUS MODEL TO GET A AND SUBSEQUENTLY USING THE RESIDUAL TO GET B
            %  A_no_X, B_no_X (from VARX without X)
            [ A_no_X_i,B_no_X_i ] = extract_AB(Betas_no_X( :,:,i), p, k, q);
            y_diff = f_irf_X(j_ex,shock_size,hor,A_no_X_i,[],B_no_X_i);
            nexttile(tilenum(tiles,2,i))
            plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
            if i==1, ylabel(sprintf('VAR A,B',nb)), end

            %  A_KFs_no_X, B_KFs_no_X (from KF without X)
            A_no_X_i = A_KFs_no_X( :,:,:,i); B_no_X_i = B_KFs_no_X( :,:,i);
            y_diff = f_irf_X(j_ex,shock_size,hor,A_no_X_i,[],B_no_X_i);
            nexttile(tilenum(tiles,3,i))
            plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
            if i==1, ylabel(sprintf('KF A, VAR B',nb)), end
        end

        %% KF IN SVD SPACE: A_KFVs based on n_SVD1
        %  A_KFVs, B_KFVs already transformed back to the original space in f_KF()
        % beta_stacked_mean is already added back to beta_filts, A_KFVs, B_KFVs
        %   in f_results > f_KF (when the SVD matrices are passed)
        % y_diff = f_irf_X(j_ex,shock_size,hor,A_KFs( :,:,1,i),[],B_KFs( :,:,i),[],[],[],0);
        y_diff = f_irf_X(j_ex,shock_size,hor,A_KFVs(:,:,1,i),[],B_KFVs(:,:,i));
        if flag_plot_all
            nexttile(tilenum(tiles,4,i))
        else
            nexttile
        end
        plot(hor_plot,y_diff(hor_plot,:),'LineWidth',1.5), title(countries{i})
        if i==1, ylabel(sprintf('KF in SVD space, n_{SVD1} = %i',n_SVD1)), end
        y_diffs(i,:) = y_diff(2,:);
    end

    for i=1:n_countries
        fprintf('%.3f,%.3f,%.3f,%.3f,%s\n',y_diffs(i,:),countries{i})
    end
end

keyboard, return

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

if isempty(country)    % if no country, then return with the list of countries
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

P00            = P0;          % from f_P0
Beta00         = Beta0;       % simple initial value (scaled identity)
ts             = dates.quarter;
dts            = ts(2:end);

P0             = 0.1*P00;
Beta0          = Beta00;
R_scale0       = 0.25;
Q              = 0.1*P00;

n_iter = 10;
R_scale = R_scale0 * Sigma_u;

flag_structural_shock = 1;
if flag_structural_shock
    ts_change = [94 104];
    R_size    = 1000;
    R_change  = @(t) 1 + (R_size - 1)*(t>ts_change(1) && t<ts_change(2));
%   Q_change  = Q      *1000;
%   Q_change(k+1:n:end,k+1:n:end) = Q(k+1:n:end,k+1:n:end);
else
    ts_change = []; R_change = []; % Q_change = [];
end

% Full sized Beta0,P0,Q
% First time      - no SVD
% Subsequent time -    SVD

%% EM Algorithm to estimate Q, R_scale
[Beta0,P0,Q,R_scale,loglik_hist,Qs,R_scales] = ...
    f_EM_KF_Q2(x,y,Beta0,P0,Q,Sigma_u,R_scale,p,n_iter,...
    V_stacked,n_SVD,beta_mean,ts_change,R_change);

%% Final KF + smoother to get final results for beta (beta_filt_test)
% Full sized Beta0,P0,Q
%            beta_filt_test,beta_smooth,P_filt0,P_smooth0
% f_KF_smoother_Q() is also called inside f_EM_KF_Q2()
flag_V = 0;  % full analysis
[beta_filt_test,beta_smooth,P_filt0,P_smooth0, J_all,P_lag, v_all,S_all0,innov_score, ...
      dbeta_smooth,smooth_filt_gap] = f_KF_smoother_Q(x,y,Beta0,P0,Q,Sigma_u,R_scale,p,...
      flag_V,V_stacked,n_SVD,beta_mean,ts_change,R_change);

[ A_KF,B_KF ] = extract_AB(beta_filt_test(:,T), p, k, q);
beta_filt     = beta_filt_test;

if flag_structural_shock
    fig = figure(96); fig.Name = 'Increased R';
    plot(ts,beta_filt_test(k+1:n:end,:)','-' ), hold on, ax = gca; ax.ColorOrderIndex = 1;
    plot(ts,beta_smooth(   k+1:n:end,:)','--'), hold off
    title(sprintf('Beta %s',X_names{1})), grid on
    keyboard
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

% Asian crisis            1997–1998
% China WTO entry         2001
% Global financial crisis 2008–2009
% Commodity collapse      2014–2016
% COVID                   2020

Country  = f_country_name(country);
ns       = 1:nb;

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
    x,y,Beta0_full,P0_full,Q0_full,Sigma_u,R_scale,p,n_iter,V,n_SVD,beta_mean0_full,...
    ts_change,R_change)

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
        f_KF_smoother_Q(x,y,beta_pred,P_pred,Q, Sigma_u,R_scale,p,flag_V,V,n_SVD,beta_mean,...
                        ts_change,R_change);
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
            H = H * V;              % (k,n_SVD)
        end
        yt = y(t,:)';
        bt = beta_smooth(:,t);

        vs = yt - H * bt;           % innovation = v = r + H*b 
        R_add     = (vs*vs') + H * P_smooth(:,:,t) * H';
        count_add = 1;
        try
            R_add = R_add/R_change(t);
            count_add = 1/R_change(t);
        catch ME
        end
        R_new = R_new + R_add;
        count = count + count_add;
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


function [beta_filt,beta_smooth,P_filt,P_smooth,J_all,P_lag,v_all,S_all,innov_score, ...
          dbeta_smooth,smooth_filt_gap] = ...
    f_KF_smoother_Q(x,y,Beta0,P0,Q,Sigma_u,R_scale,p,flag_V,V,n_SVD,beta_mean0,...
                    ts_change,R_change)

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
        flag_SVD    = true;
        beta_mean   = V'*beta_mean0(:);  % V subspace (beta_mean), whole space (beta_mean0)
        beta_pred_t = V'*Beta0(:);       % use V subspace and convert back to whole space later
        P_pred_t    = V'*P0*V;
        Q           = V'*Q *V;
    else
        flag_SVD    = false;
        beta_mean   = beta_mean0(:);
        beta_pred_t = Beta0(:);
        P_pred_t    = P0;    % your diagonal prior
    end
else
    flag_SVD  = false;
    n         = k*p + q;
    nb        = n*k;
    beta_mean = 0;
    beta_pred_t = Beta0(:);
    P_pred_t    = P0;    % your diagonal prior    
end

%% ORIGINAL
% beta_pred_t = Beta0(:);
% P_pred_t    = P0;

%% CONTINUE WITH nb
beta_filt = zeros(nb, T);
P_filt    = zeros(nb, nb, T);
I         = eye(nb);

% --- store predicted quantities for smoother
beta_pred = zeros(nb, T);
P_pred    = zeros(nb, nb, T);

% --- store smoother gain
J_all     = zeros(nb, nb, T);

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
beta_filt(    :,1) = beta_pred_t; % Beta0(:);
beta_pred(:,1) = beta_pred_t; % Beta0(:);
P_filt(    :,:,1)  = P_pred_t;    % P0;
P_pred(:,:,1)  = P_pred_t;    % P0;

%%    Kalman filter
for t = 2:T
    % --- Prediction step
    Qt = Q; % default
    Rt = R;
    try Rt = R_change(t)*R; catch ME, end
    P_pred_t = P_pred_t + Qt;

    % --- Store predicted quantities
    beta_pred(:,t) = beta_pred_t;
    P_pred( :,:,t) = P_pred_t;

    % --- Measurement
    % ZVt = Zt*V
    Zt    = build_Zt(y, x, t, p);      % (k,   nb_orig), time t-1 ... t-p
    if flag_V                          % both 1 or 2
        Zt = Zt * V;                   % (k,n_SVD)
    end
    H     = Zt;

    % --- Innovation
    yt    = y(t,:)';
%   v     = yt - H * beta_pred_t;
    v     = yt - H * (beta_pred_t + beta_mean);

    % --- Innovation covariance
    %     y = H*b + r  = H*(b_pred + b_err) + r =  H*b_pred + v
    %     y - H*b_pred = H*b_err + r = v = innovation
    S = H * P_pred_t * H' + Rt;                 % Rt large > S large

    % --- Store innovation diagnostics
    v_all(:,t)     = v;
    S_all(:,:,t)   = S;
    innov_score(t) = real(v' * (S \ v));

    % --- Kalman gain
    K = P_pred_t * H' / S;                      % S large > K small

    % --- Update
    beta_filt(:,t) = beta_pred_t + K * v;       % K small > beta_filt(:,t) approx (:,t-1)
    KH             = K * H;
    P_filt(:,:,t)  = (I - KH) * P_pred_t * (I - KH)' + K * Rt * K'; % P_filt approx P_pred

    % --- Prepare next iteration
    beta_pred_t = beta_filt(:,t);               % beta_pred(:,t+1) approx beta_filt(:,t)
    P_pred_t    = P_filt(:,:,t);                % P_pred(   :,t+1) approx P_filt(   :,t)
end

%%    RTS smoother
%     No need to inlcude changes in Q and R because they are already implicitly included
%       in the KF matrices that are passed on to the smoother
beta_smooth = zeros(nb, T);
P_smooth    = zeros(nb, nb, T);

% --- Start backward pass from final filtered estimate
beta_smooth(:,T) = beta_filt(:,T);
P_smooth( :,:,T) = P_filt( :,:,T);

for t = T-1:-1:1
    % Since F = I:
    J            = P_filt(:,:,t) / P_pred(:,:,t+1); % Rt large > J approx I
    J_all(:,:,t) = J;

    % --- Smoothed state                              Rt large > beta_smooth unchanged
    beta_smooth(:,t) = beta_filt(:,t) + J*(beta_smooth(:,t+1) - beta_pred(:,t+1));

    % --- Smoothed covariance                         Rt large > P_smooth unchanged
    P_smooth( :,:,t) = P_filt( :,:,t) + J*(P_smooth( :,:,t+1) - P_pred( :,:,t+1))*J';
end

%% Lag-one smoothed covariance for EM
%  Initialization at final time
%  P_lag(:,:,T) = Cov(beta_T, beta_{T-1} | Y)

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


function [ y_diff, y_diff_KF ] = f_irf_X(j_ex,shock_size,hor,A,A_KF,B,B_KF,flag_plot,fig_no, ...
    flag_ENSO)

[ ~,q   ] = size(B);
[ ~,k,p ] = size(A);

if q>=max(j_ex)                     % run only if the exogenous variable exists
    x_base          = zeros(hor,q); % rows are time
    x_shock         = x_base;

    try flag_ENSO; catch, flag_ENSO = 0; end

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
