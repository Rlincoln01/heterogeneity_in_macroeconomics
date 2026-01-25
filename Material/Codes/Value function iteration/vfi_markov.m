%=========================================================================
%                     Value function Iteration method
%=========================================================================

% clear workspace
clear;
close all;

%% Set Parameters

% Household Parameters
beta = 0.99;
r = 0.032;

% Capital Accumulation
delta = 0.025;

% return on asset
R = 1 + r -delta; %gross return on capital

% Labor Market
w = 2.55;
l_bar = 1/0.9; % # of hours worked
tau = 0.015; % tax on earnings
mu = 0.15; % share of wage received as unemployment benefit

% preferences
u = @(c)log(c);

% Initial conditions
borrow_lim  = 0; % ad-hoc borrowing constraint

%% Stochastic settings
% Markov Chains
p_00 = 0.6; % probability of remaining unemployed
p_01 = 1-p_00; % probability of being employed
p_11 = 0.955555; % probability of remaining employed
p_10 = 1 - p_11; % probability of being fired

P = [p_00 p_01; p_10 p_11]; % transition matrix




%% Grid

% set capital grids
na          = 500; % Number of points on the grid 
amax        = 20; % max # of assets
agrid_par   = 1; %1 for linear, 0 for L-shaped (almost all points near 0)

agrid = linspace(0,1,na)'; 
agrid = agrid.^(1./agrid_par);
agrid = borrow_lim + (amax-borrow_lim).*agrid; %operations are done elementwise


% Income grid
ny = 2; % n of points in the grid
y = @(e) e*(1-tau)*l_bar*w + (1-e)*mu*w; % returns labor income given if unemployed or no
ygrid = [y(0) y(1)];

%% Initialize on value function

% guess of the value function
Vguess = zeros(na,ny);
for ie = 1:ny
    Vguess(:,ie) = u((r-delta).*agrid + ygrid(ie))./(1-beta); 
end
%% Computation

% computation parameters
max_iter    = 2000;
tol_iter    = 1.0e-6;
Nsim        = 50000;
Tsim        = 500;

V=Vguess;

Vdiff = 1;
iter = 0;

%% OPTIONS
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;

while iter <= max_iter && Vdiff>tol_iter
    iter = iter + 1;
    Vlast = V;
    V = zeros(na,ny);
    cap = zeros(na,ny);
    capind = zeros(na,ny);
    con = zeros(na,ny);

    % Loop over Capital
    for ik = 1:na 
        % Loop over income
        for ie = 1:ny
            % determine the distribution of income
            if ygrid(ie) == y(0) % if he's unemployed
                p = [1 0];
            elseif ygrid(ie) == y(1) % if he's employed
                p = [0 1];
            end
            % solve the maximinization problem
            dist = transpose(P)*transpose(p); %conditional distribution
            cash = R.*agrid(ik) + ygrid(ie); 
            Vchoice = u(max(cash-agrid,1.0e-10)) + beta.*(Vlast*dist);
            [V(ik,ie),capind(ik,ie)] = max(Vchoice);
            cap(ik,ie) = agrid(capind(ik,ie));
            con(ik,ie) = cash - cap(ik,ie);
        end
    end

    Vdiff = max(max(abs(V-Vlast)));
    if Display >=1
        disp(['Iteration no. ' int2str(iter), ' max val fn diff is ' num2str(Vdiff)]);
    end
end

%% Simulation

% Stationary distribution of the Markov Chain
yasy = asymptotics(dtmc(P));
ycumdist = cumsum(yasy);

%Draw random numbers
rng(2017)

if DoSimulate ==1
    yindsim = zeros(Nsim,Tsim);
    aindsim = zeros(Nsim,Tsim);
    

    % initial assets
    aindsim(:,1) = 1;

    % initial income distribution ~ Asymptotic MC distribution %
    yrand = rand(Nsim,1);

    for nsim = 1:Nsim
        if yrand(nsim) >= ycumdist(1)
            s0 = 2;
        else
            s0 =1;
        end
        yindsim(nsim,:) = simulateMC([1 2]', P, "size", Tsim, "statezero",s0);
    end
    disp("MC simulated for each individual")

    % loop over time periods t>1
    for it = 1:Tsim
        if Display >=1 && mod(it,100) ==0
             disp([' Simulating, time period ' int2str(it)]);
        end

        % asset choice
        if it<Tsim
            for iy = 1:ny
                aindsim(yindsim(:,it)==iy,it+1) = capind(aindsim(yindsim(:,it)==iy,it),iy);
            end
        end
    end

    %assign actual asset and income values;
    asim = agrid(aindsim);
    ysim = ygrid(yindsim);
end








%% MAKE PLOTS
if MakePlots ==1 
    
    ff = figure(1);
    ff.Position = [50 50 250*4 250*2.5];   
    
    % consumption policy function
    subplot(2,2,1);
    plot(agrid,con(:,1),'b-',agrid,con(:,ny),'r-','LineWidth',1);
    grid;
    xlim([0 amax]);
%     title('Consumption Policy Function');
    title('Consumption');
    legend('Unemployed','Employed','location','north');

    % savings policy function
    subplot(2,2,2);
    plot(agrid,cap(:,1)-agrid,'b-',agrid,cap(:,ny)-agrid,'r-','LineWidth',1);
    hold on;
    plot(agrid,zeros(na,1),'k','LineWidth',0.5);
    hold off;
    grid;
    xlim([0 amax]);
%     title('Savings Policy Function (a''-a)');
    title('Savings');
    
              
    %asset distribution
    subplot(2,2,3);
    histogram(asim(:,Tsim),0:1:amax);
    h = findobj(gca,'Type','patch');
    set(h,'FaceColor',[.7 .7 .7],'EdgeColor','black','LineStyle','-');
    ylabel('')
    title('Asset distribution');

    %convergence check
    subplot(2,2,4);
    plot((1:Tsim)',mean(asim,1),'k-','LineWidth',1.5);
    ylabel('Time Period');
    title('Mean Asset Convergence');
    
    
   % asset distribution statistics
    aysim = asim(:,Tsim) ./ mean(ysim(:,Tsim));
    disp(['Mean assets: ' num2str(mean(aysim))]);
    disp(['Fraction borrowing constrained: ' num2str(sum(aysim==borrow_lim)./Nsim * 100) '%']);
    disp(['10th Percentile: ' num2str(quantile(aysim,.1))]);
    disp(['50th Percentile: ' num2str(quantile(aysim,.5))]);
    disp(['90th Percentile: ' num2str(quantile(aysim,.9))]);
    disp(['99th Percentile: ' num2str(quantile(aysim,.99))]);

end








