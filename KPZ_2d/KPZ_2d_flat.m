clear; clc; close all;

%% =========================================================
%% 2D KPZ EQUATION ON A FLAT DOMAIN WITH PBC
%%
%% Governing equation:
%%
%% dh/dt =
%%
%% nu * nabla^2 h
%% + (lambda/2) * |grad h|^2
%% - kappa * h
%% + eta(x,y,t)
%%
%% on a periodic 2D square domain.
%%
%% =========================================================

%%=========================================================
%%-- PARAMETERS
%%=========================================================

%%---- physics

nu       = 1.0;      % diffusion / smoothing
lambda   = 2.0;      % KPZ nonlinearity
kappa    = 0.0;      % restoring confinement
noiseAmp = 1.0;      % noise strength

%%---- optional mean growth velocity

v0 = 0.0;

%%=========================================================
%%--- NUMERICS
%%=========================================================

Nx = 128;
Ny = 128;

Lx = 100;
Ly = 100;

dx = Lx/Nx;
dy = Ly/Ny;

Nt = 10000;

dt = 1e-3;

%%=========================================================
%%--- GRID
%%=========================================================

x = linspace(0,Lx,Nx);
y = linspace(0,Ly,Ny);

[X,Y] = meshgrid(x,y);

%%=========================================================
%%--- INITIAL CONDITION
%%=========================================================

%%---- small random fluctuations

h = 0.01*randn(Ny,Nx);

%%=========================================================
%%--- STORAGE
%%=========================================================

W = zeros(Nt,1);

hmean = zeros(Nt,1);

time = (0:Nt-1)*dt;

%%=========================================================
%%--- MAIN LOOP
%%=========================================================

for it = 1:Nt

    disp(it)

    %%=====================================================
    %%--- PERIODIC NEIGHBORS
    %%=====================================================

    hL = circshift(h,[0 +1]);
    hR = circshift(h,[0 -1]);

    hU = circshift(h,[+1 0]);
    hD = circshift(h,[-1 0]);

    %%=====================================================
    %%--- FIRST DERIVATIVES
    %%=====================================================

    dhdx = (hR - hL)/(2*dx);

    dhdy = (hD - hU)/(2*dy);

    %%=====================================================
    %%--- SECOND DERIVATIVES
    %%=====================================================

    d2hdx2 = (hR - 2*h + hL)/(dx^2);

    d2hdy2 = (hD - 2*h + hU)/(dy^2);

    %%=====================================================
    %%-- LAPLACIAN
    %%--
    %%-- nabla^2 h
    %%=====================================================

    LapH = d2hdx2 + d2hdy2;

    %%=====================================================
    %%-- KPZ NONLINEAR TERM
    %%--
    %%-- |grad h|^2
    %%=====================================================

    grad2 = dhdx.^2 + dhdy.^2;

    %%=====================================================
    %%-- STOCHASTIC NOISE
    %%--
    %%-- Correct Ito scaling:
    %%--
    %%-- eta ~ sqrt(dt)
    %%=====================================================

    eta = (noiseAmp/sqrt(dt))*randn(Ny,Nx);

    %%=====================================================
    %%-- EXPLICIT EULER UPDATE
    %%=====================================================

    h_new = h + dt*(...
        v0 ...
        + nu*LapH ...
        + 0.5*lambda*grad2 ...
        - kappa*h ...
        + eta);

    %%=====================================================
    %%-- OPTIONAL STABILITY CLIPPING
    %%=====================================================

   % h_new(abs(h_new)>50) = 50;

    h = h_new;

    %%=====================================================
    %%-- MEAN HEIGHT
    %%=====================================================

    hbar = mean(h(:));

    hmean(it) = hbar;

    %%=====================================================
    %%-- INTERFACE ROUGHNESS
    %%--
    %%-- W(t) = sqrt( <(h-hbar)^2> )
    %%=====================================================

    W(it) = sqrt(...
        mean((h(:)-hbar).^2));

    %%=====================================================
    %%-- VISUALIZATION
    %%=====================================================

    % if mod(it,1)==0
    % 
    %     figure(1)
    %     clf
    % 
    %     surf(X,Y,h,...
    %         'EdgeColor','none')
    % 
    %     shading interp
    % 
    %     view(2)
    % 
    %     axis equal tight
    % 
    %     colorbar
    % 
    %     title(['t = ',num2str(it*dt)])
    % 
    %     xlabel('x')
    %     ylabel('y')
    % 
    %     set(gca,...
    %         'FontSize',18,...
    %         'LineWidth',2)
    % 
    %     drawnow
    % 
    % end

end

%%=========================================================
%%-- SMOOTH ROUGHNESS
%%=========================================================

W_smooth = sgolayfilt(W,3,31);

hmean_smooth = sgolayfilt(hmean,3,31);

%% =========================================================
%% PLOT : W(t)
%% =========================================================

figure('Color','w')

loglog(time,W,...
    'Color',[0.7 0.7 0.7],...
    'LineWidth',2, 'HandleVisibility','off')

hold on

loglog(time,W_smooth,...
    '-r',...
    'LineWidth',4, 'HandleVisibility','off')

hold on 
pow = 1/2; 
loglog(time, time.^(pow), ':k', 'LineWidth',2, 'DisplayName',strcat('t^{', num2str(pow), '}'))
pow = 1/3; 
loglog(time, 0.5*time.^(pow), '--k', 'LineWidth',2, 'DisplayName',strcat('t^{', num2str(pow), '}'))

xlabel('t')

legend('Location','northwest', 'FontSize',24)

ylabel('W(t)')

axis square

set(gca,...
    'FontSize',22,...
    'LineWidth',2)

%% =========================================================
%% PLOT : W / mean height
%% =========================================================

% figure('Color','w')
% 
% loglog(time,...
%     W./abs(hmean),...
%     '-b',...
%     'LineWidth',4)
% 
% xlabel('Time')
% 
% ylabel('W / <h>')
% 
% axis square
% 
% set(gca,...
%     'FontSize',22,...
%     'LineWidth',2)

%% =========================================================
%% PLOT : W vs mean height
%% =========================================================

% figure('Color','w')
% 
% loglog(abs(hmean_smooth),...
%     W_smooth,...
%     '-k',...
%     'LineWidth',4)
% 
% xlabel('<h>')
% 
% ylabel('W')
% 
% axis square
% 
% set(gca,...
%     'FontSize',22,...
%     'LineWidth',2)

%% =========================================================
%% FINAL HEIGHT SNAPSHOT
%% =========================================================

% figure('Color','w')
% 
% surf(X,Y,h,...
%     'EdgeColor','none')
% 
% shading interp
% 
% axis equal tight
% 
% xlabel('x')
% ylabel('y')
% zlabel('h')
% 
% title('Final interface')
% 
% colorbar
% 
% view(45,30)
% 
% set(gca,...
%     'FontSize',20,...
%     'LineWidth',2)

%% =========================================================
%% SUGGESTED PARAMETER REGIMES
%% =========================================================

%% ---- Edwards-Wilkinson
%%
%% nu = 1
%% lambda = 0
%% kappa = 0
%% noiseAmp = 1

%% ---- Weak KPZ
%%
%% nu = 1
%% lambda = 0.5
%% kappa = 0
%% noiseAmp = 1

%% ---- Strong KPZ
%%
%% nu = 1
%% lambda = 3
%% kappa = 0
%% noiseAmp = 1
%%
%% may require dt = 1e-4

%% ---- Confinement-dominated
%%
%% nu = 1
%% lambda = 1
%% kappa = 1
%% noiseAmp = 1

%% =========================================================
%% NUMERICAL NOTES
%% =========================================================

%% Explicit Euler stability roughly requires:
%%
%% dt < dx^2 / nu
%%
%% Large lambda can produce blow-up.
%%
%% If unstable:
%%
%% - reduce dt
%% - reduce lambda
%% - increase nu
%%
%% Spectral methods are much better for large systems.
%%
%% =========================================================
