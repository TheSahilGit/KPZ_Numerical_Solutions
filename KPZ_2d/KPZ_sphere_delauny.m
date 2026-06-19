clear; clc; close all;

%% =========================================================
%% STOCHASTIC KPZ-TYPE EQUATION ON A TRIANGULATED SPHERE
%%
%% Surface:
%%
%% R_i(t) = R0 + h_i(t)
%%
%% defined on vertices of a spherical triangulated mesh.
%%
%% Evolution:
%%
%% dh_i/dt =
%%
%%     nu * Delta_S h_i
%%   + (lambda/2) * |grad_S h_i|^2
%%   - kappa * h_i
%%   + eta_i
%%
%% using:
%%
%% 1. Cotangent Laplace-Beltrami operator
%% 2. Piecewise-linear FEM gradient on triangles
%%
%%=========================================================
%%=========================================================
%%-- PARAMETERS
%%=========================================================

R0 = 1;

nu       = 0.1;
lambda   = 0.5; %0.05;
kappa    = 0;

noiseAmp = 0.5;


Nt = 1000000;

dt   = 1e-5;



%% =========================================================
%% SPHERE MESH : FIBONACCI SPHERE + CONVEX HULL
%% =========================================================

Npts = 256;

%%---- Fibonacci sphere points

i = (0:Npts-1)';

golden_angle = pi*(3 - sqrt(5));

phi = golden_angle * i;

z = 1 - 2*(i + 0.5)/Npts;

r = sqrt(1 - z.^2);

x = r .* cos(phi);
y = r .* sin(phi);

%% ---- vertex positions on unit sphere

V = [x y z];

%% =========================================================
%% TRIANGULATION
%%
%% Convex hull of points on sphere gives
%% spherical triangulation.
%% =========================================================

tri = convhull(V);

%%---- sizes

Nv = size(V,1);

Ntmesh = size(tri,1);

%%=========================================================
%%INITIAL CONDITION
%%=========================================================

h = 0.05*randn(Nv,1);

%% =========================================================
%% BUILD CONNECTIVITY
%% =========================================================

neighbors = cell(Nv,1);

for t = 1:Ntmesh

    f = tri(t,:);

    i = f(1);
    j = f(2);
    k = f(3);

    neighbors{i} = unique([neighbors{i}, j, k]);
    neighbors{j} = unique([neighbors{j}, i, k]);
    neighbors{k} = unique([neighbors{k}, i, j]);

end

%% =========================================================
%% VORONOI AREA
%%
%% barycentric area approximation
%% =========================================================

Ai = zeros(Nv,1);

for t = 1:Ntmesh

    ids = tri(t,:);

    p1 = V(ids(1),:);
    p2 = V(ids(2),:);
    p3 = V(ids(3),:);

    At = 0.5*norm(cross(p2-p1,p3-p1));

    Ai(ids) = Ai(ids) + At/3;

end

%%=========================================================
%%-- PRECOMPUTE COTANGENT WEIGHTS
%%=========================================================

Wcot = sparse(Nv,Nv);

for t = 1:Ntmesh

    ids = tri(t,:);

    i = ids(1);
    j = ids(2);
    k = ids(3);

    vi = V(i,:);
    vj = V(j,:);
    vk = V(k,:);

    %% ---- cotangent opposite edge ij

    cot_k = Cotangent(vi-vk,vj-vk);

    %% ---- cotangent opposite edge jk

    cot_i = Cotangent(vj-vi,vk-vi);

    %% ---- cotangent opposite edge ki

    cot_j = Cotangent(vk-vj,vi-vj);

    %% ---- accumulate symmetric weights

    Wcot(i,j) = Wcot(i,j) + cot_k;
    Wcot(j,i) = Wcot(j,i) + cot_k;

    Wcot(j,k) = Wcot(j,k) + cot_i;
    Wcot(k,j) = Wcot(k,j) + cot_i;

    Wcot(k,i) = Wcot(k,i) + cot_j;
    Wcot(i,k) = Wcot(i,k) + cot_j;

end

%% =========================================================
%% STORAGE
%% =========================================================

Wrough = zeros(Nt,1);

Rmean = zeros(Nt,1);

time = (0:Nt-1)*dt;

%% =========================================================
%% MAIN LOOP
%% =========================================================


outdir = 'frames';

if ~exist(outdir,'dir')
    mkdir(outdir);
end


for it = 1:Nt

    disp(it)

    %% =====================================================
    %% LAPLACE-BELTRAMI
    %%
    %% Delta h_i =
    %%
    %% (1/(2Ai))
    %% sum_j wij (hj-hi)
    %%
    %% =====================================================

    LapH = zeros(Nv,1);

    for i = 1:Nv

        Ni = neighbors{i};

        s = 0;

        for jj = 1:length(Ni)

            j = Ni(jj);

            wij = Wcot(i,j);

            s = s + wij*(h(j)-h(i));

        end

        LapH(i) = s/(2*Ai(i));
        %LapH(i) = mean(h(Ni)) - h(i);

    end

    %% =====================================================
    %% GRADIENT TERM
    %%
    %% Piecewise FEM gradient on triangles
    %%
    %% =====================================================

    grad2_vertex = zeros(Nv,1);

    area_accum = zeros(Nv,1);

    for t = 1:Ntmesh

        ids = tri(t,:);

        i = ids(1);
        j = ids(2);
        k = ids(3);

        xi = V(i,:);
        xj = V(j,:);
        xk = V(k,:);

        fi = h(i);
        fj = h(j);
        fk = h(k);

        %% ---- triangle normal

        Nvec = cross(xj-xi,xk-xi);

        At = 0.5*norm(Nvec);

        nHat = Nvec/(norm(Nvec)+1e-12);

        %%=================================================
        %% DISCRETE FEM GRADIENT
        %%
        %% grad f =
        %%
        %% (fj-fi)*(xi-xk)^perp /(2At)
        %% +
        %% (fk-fi)*(xj-xi)^perp /(2At)
        %%
        %% =================================================

        e1 = cross(nHat,xi-xk);
        e2 = cross(nHat,xj-xi);

        gradf = ...
            (fj-fi)*e1/(2*At) ...
            + (fk-fi)*e2/(2*At);

        g2 = dot(gradf,gradf);

        %% ---- distribute to vertices

        grad2_vertex(i) = grad2_vertex(i) + g2*At/3;
        grad2_vertex(j) = grad2_vertex(j) + g2*At/3;
        grad2_vertex(k) = grad2_vertex(k) + g2*At/3;

        area_accum(i) = area_accum(i) + At/3;
        area_accum(j) = area_accum(j) + At/3;
        area_accum(k) = area_accum(k) + At/3;

    end

    %% ---- area averaged gradient

    grad2_vertex = grad2_vertex ./ (area_accum + 1e-12);

    %% =====================================================
    %% STOCHASTIC NOISE
    %% =====================================================

    %eta = (noiseAmp./sqrt(dt)) .* randn(Nv,1);
    eta = (noiseAmp./sqrt(Ai*dt)) .* randn(Nv,1);

    %% =====================================================
    %% EXPLICIT EULER UPDATE
    %% =====================================================

    h_new = h + dt*(...
        nu*LapH ...
        + 0.5*lambda*grad2_vertex ...
        - kappa*h ...
        + eta);

    %% ---- optional clipping

    % h_new(abs(h_new)>5*R0) = 5*R0;

    h = h_new;

    %% =====================================================
    %% MEAN RADIUS (activate if domain growth is needed. )
    %% =====================================================

    % R = R0 + h;
    %
    % Rbar = sum(R .* Ai) / sum(Ai);
    %
    % Rmean(it) = Rbar;



    %% =====================================================
    %% ROUGHNESS
    %%
    %% W = sqrt(<(R-Rbar)^2>)
    %% =====================================================
    %
    % Wrough(it) = sqrt(...
    %     sum(Ai .* (R-Rbar).^2) / sum(Ai));  -- Activate if domain growth.

    hbar = sum(h .* Ai) / sum(Ai);

    Wrough(it) = sqrt(...
        sum(Ai .* (h-hbar).^2) / sum(Ai));

    %% =====================================================
    %% VISUALIZATION
    %% =====================================================

    % if mod(it,2)==0
    % 
    %     Rplot = R0 ;
    % 
    %     X = Rplot .* V(:,1);
    %     Y = Rplot .* V(:,2);
    %     Z = Rplot .* V(:,3);
    % 
    %     figure(1)
    %     clf
    % 
    %     trisurf(tri,...
    %         X,Y,Z,...
    %         h,...
    %         'EdgeColor','k');
    % 
    %     axis equal
    % 
    %     xlabel('x')
    %     ylabel('y')
    %     zlabel('z')
    % 
    %     title(['t = ',num2str(it*dt)])
    % 
    %     clim([-15 15])
    % 
    %     cb = colorbar;
    % 
    %     cb.Label.String = '$h(\theta, \phi)$';
    %     cb.Label.Interpreter = 'latex';
    % 
    % 
    %     cb.Label.FontSize = 28;
    % 
    %     cb.Label.Rotation = 270;   % optional
    % 
    %     cb.Label.VerticalAlignment = 'bottom';
    % 
    %     view(3)
    % 
    %     camlight
    %     lighting gouraud
    % 
    %     set(gca,...
    %         'FontSize',18,...
    %         'LineWidth',2)
    % 
    %     xticks([])
    %     yticks([])
    %     zticks([])
    %     xlabel("")
    %     ylabel("")
    %     zlabel("")
    %     box on
    % 
    %     drawnow
    % 
    %     fname = sprintf('%s/frame_%05d.png',outdir,it);
    % 
    %     exportgraphics(gcf,fname,'Resolution',100);
    % 
    % end

end

%% =========================================================
%% SMOOTH DATA
%% =========================================================

W_smooth = sgolayfilt(Wrough,3,31);

Rmean_smooth = sgolayfilt(Rmean,3,31);

%% =========================================================
%% PLOT : W(t)
%% =========================================================

figure('Color','w')

% loglog(time,Wrough,...
%     'Color',[0.7 0.7 0.7],...
%     'LineWidth',2)

%hold on

loglog(time,Wrough,...
    '-r',...
    'LineWidth',4)
hold on

loglog(time, 3*time.^(1/3), ':k', 'LineWidth',3)

xlabel('t')

ylabel('W(t)')

axis square

set(gca,...
    'FontSize',22,...
    'LineWidth',2)

%% =========================================================
%% PLOT : W/R
%% =========================================================

% figure('Color','w')
%
% loglog(time,...
%     W_smooth./Rmean_smooth,...
%     '-b',...
%     'LineWidth',4)
%
% xlabel('t')
%
% ylabel('W/R')
%
% axis square
%
% set(gca,...
%     'FontSize',22,...
%     'LineWidth',2)

%% =========================================================
%% PLOT : W vs R
%% =========================================================

% figure('Color','w')
%
% loglog(Rmean_smooth,...
%     W_smooth,...
%     '-k',...
%     'LineWidth',4)
%
% xlabel('R')
%
% ylabel('W')
%
% axis square
%
% set(gca,...
%     'FontSize',22,...
%     'LineWidth',2)

%% =========================================================
%% FUNCTION : COTANGENT
%% =========================================================

function c = Cotangent(a,b)

c = dot(a,b)/(norm(cross(a,b))+1e-12);

end