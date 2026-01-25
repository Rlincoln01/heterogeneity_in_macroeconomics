function y=simulateMC(z,P,varargin)
% simulateMC: Given a vector of states (z) and a transition matrix (P), the
% function generates possible paths of a Markov process.
% 
% Syntax
% =========
% y = simulateMC(z,P,ops)
%
% Options
% =========
% 'size'       : The size of the generated path, T-1. (Default = 100)
% 'statezero'  : The initial state of the system (Default= 1)
% 'random'     : Vector for pseudo random numbers in (0,1).
%
% Outputs
% =========
% y : Path of size 'T-1' (the first element is the initial).
% ===========================================================================
% By Alex Carrasco - March, 2017
% ===========================================================================

% Defaults
T=1000;
init=0;
u=rand(T,1);
i=1;
flaginit=0;

% Options
for ii=1:numel(varargin)
    if strcmp(varargin{ii},'size')      ; T=varargin{ii+1}; end
    if strcmp(varargin{ii},'statezero') ; i=varargin{ii+1}; end
    if strcmp(varargin{ii},'initial')   ; init=varargin{ii+1}; flaginit=1; end
    if strcmp(varargin{ii},'random')    ; u=varargin{ii+1}; end
end

% Simulation
y    = nan(T,1);

if flaginit
    aux  = z(init>z);
    y(1) = aux(end);
    i    = numel(aux);  
else
    y(1)=z(i);
end

for t = 2:T
   P_0 = cumsum(P(i,:));
   if P_0(1)>u(t), j=1; else j=sum(P_0<u(t))+1; end
   y(t)=z(j);
   i=j;
end

end