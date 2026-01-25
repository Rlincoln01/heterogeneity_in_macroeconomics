%=========================================================================
%                     Endogenous Grid Point Method
%=========================================================================

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
u = @(c)log(c); %utility function

u1 = @(c)1./c; % first derivative of the utility function

u1inv = @(u) 1./u; % inverse of the first derivative of the utility function

% Initial conditions
borrow_lim  = 0; % ad-hoc borrowing constraint


%% Set up  grids

% asset grids
na          = 30;
amax        = 30; 
agrid_par   = 0.4; %1 for linear, 0 for L-shaped
agrid = linspace(0,1,na)';
agrid = agrid.^(1./agrid_par);
agrid = borrow_lim + (amax-borrow_lim).*agrid;

% income: markov chain with employment
p_00 = 0.6; % probability of remaining unemployed
p_01 = 1-p_00; % probability of being employed
p_11 = 0.955555; % probability of remaining employed
p_10 = 1 - p_11; % probability of being fired

P = [p_00 p_01; p_10 p_11]; % transition matrix

% Income grid
ny = 2; % n of points in the grid
y = @(e) e*(1-tau)*l_bar*w + (1-e)*mu*w; % returns labor income given if unemployed or no
ygrid = [y(0) y(1)];



%% Computation parametrization

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 50000;
Tsim        = 1000;
iter = 0;
cdiff = 1000;
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;


%% Initialize consumption function

conguess = zeros(na,ny);
for iy = 1:ny
    conguess(:,iy) = (r-delta).*agrid + ygrid(iy);
end



%% Iterate on Euler Equation with the endogenous grid point methods

con = conguess;

while iter <= max_iter && cdiff>tol_iter
    iter = iter + 1;
    conlast = con;
    sav = zeros(na,ny);
    ass1 = zeros(na,ny);
    for iy = 1:ny %fix income first in order to establish endogenous asset grid
        % determine the income distribution through a markov chain
        if ygrid(iy) == y(0) % if he's unemployed
                p = [1 0];
        elseif ygrid(iy) == y(1) % if he's employed
                p = [0 1];
        end
            ydist = transpose(P)*transpose(p); %conditional distribution
            emuc = u1(con)*ydist;
            muc1 = beta.*R.*emuc; %expected marginal utility of consumption
            con1 = u1inv(muc1); % solving for the consumption policy function (from y and a tomorrow)
            ass1(:,iy) = (con1 + agrid -ygrid(iy))./R; % asset today as a function of income and assets tomorrow
            for ia = 1:na % Partition the grid into the point above and below the optimal asset today
                if agrid(ia) < ass1(1,iy) % asset today evaluated at the future borrowing constraint
                    sav(ia,iy) = borrow_lim; % savings = assets = borroming limit
                else % if not, assets today are a linear interpolation of assets tomorrow and gridpoint
                    sav(ia,iy) = lininterp1(ass1(:,iy),agrid,agrid(ia));
                end
            end
        con(:,iy) = R.*agrid +ygrid(iy) - sav(:,iy);
    end

    cdiff = max(max(abs(con-conlast)));
    if Display >= 1 && mod(iter,20) ==0
        disp(['Iteration no. ' int2str(iter), ' max con fn diff is ' num2str(cdiff)]);
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
    asim = zeros(Nsim,Tsim);
    savinterp = cell(ny,1);

    %create interpolating function
    for iy = 1:ny
        savinterp{iy} = griddedInterpolant(agrid,sav(:,iy),'linear');
    end
    
    % initial assets
    asim(:,1) = 1;

    % initial income distribution ~ Asymptotic MC distribution %
    yrand = rand(Nsim,1);

    disp("Simulating income for each individual...")
    for nsim = 1:Nsim
        if yrand(nsim) >= ycumdist(1)
            s0 = 2;
        else
            s0 =1;
        end
        yindsim(nsim,:) = simulateMC([1 2]', P, "size", Tsim, "statezero",s0);
    end
    disp("Income simulated for each individual")

    % loop over time periods t>1
    for it = 1:Tsim
        if Display >=1 && mod(it,100) ==0
             disp([' Simulating, time period ' int2str(it)]);
        end

        % asset choice
        if it<Tsim
            for iy = 1:ny
                asim(yindsim(:,it)==iy,it+1) = savinterp{iy}(asim(yindsim(:,it)==iy,it));
            end
        end
    end

    ysim = ygrid(yindsim); % turn states into y values
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
    plot(agrid,sav(:,1)-agrid,'b-',agrid,sav(:,ny)-agrid,'r-','LineWidth',1);
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


