clear; clc; close all;

%% =========================================================
%% FAST KPZ ON TRIANGULATED SPHERE
%%
%% Explicit Euler
%% Fully vectorized
%%
%% Optimizations:
%%
%% 1. Sparse Laplace-Beltrami matrix
%% 2. Vectorized triangle gradients
%% 3. Precomputed geometry
%% 4. No inner vertex/triangle loops
%%
%% =========================================================

%% =========================================================
%% PARAMETERS
%% =========================================================

R0 = 1;

nu       = 0.1;
lambda   = 0.2;
kappa    = 0.0;

noiseAmp = 0.2;

Nt = 100000;

dt = 1e-4;

%% =========================================================
%% SPHERE MESH
%% =========================================================

Npts = 512;

i = (0:Npts-1)';

golden_angle = pi*(3 - sqrt(5));

phi = golden_angle * i;

z = 1 - 2*(i + 0.5)/Npts;

r = sqrt(1 - z.^2);

x = r .* cos(phi);
y = r .* sin(phi);

V = [x y z];

%% =========================================================
%% TRIANGULATION
%% =========================================================

tri = convhull(V);

Nv = size(V,1);

Ntmesh = size(tri,1);

%% =========================================================
%% TRIANGLE INDICES
%% =========================================================

tri_i = tri(:,1);
tri_j = tri(:,2);
tri_k = tri(:,3);

%% =========================================================
%% VORONOI / BARYCENTRIC AREA
%% =========================================================

Xi = V(tri_i,:);
Xj = V(tri_j,:);
Xk = V(tri_k,:);

Nvec = cross(Xj-Xi,Xk-Xi,2);

At = 0.5*sqrt(sum(Nvec.^2,2));

Ai = accumarray(tri_i,At/3,[Nv 1]) ...
   + accumarray(tri_j,At/3,[Nv 1]) ...
   + accumarray(tri_k,At/3,[Nv 1]);

%% =========================================================
%% COTANGENT WEIGHTS
%% =========================================================

Wcot = sparse(Nv,Nv);

for t = 1:Ntmesh

    i = tri_i(t);
    j = tri_j(t);
    k = tri_k(t);

    vi = V(i,:);
    vj = V(j,:);
    vk = V(k,:);

    cot_k = Cotangent(vi-vk,vj-vk);

    cot_i = Cotangent(vj-vi,vk-vi);

    cot_j = Cotangent(vk-vj,vi-vj);

    Wcot(i,j) = Wcot(i,j) + cot_k;
    Wcot(j,i) = Wcot(j,i) + cot_k;

    Wcot(j,k) = Wcot(j,k) + cot_i;
    Wcot(k,j) = Wcot(k,j) + cot_i;

    Wcot(k,i) = Wcot(k,i) + cot_j;
    Wcot(i,k) = Wcot(i,k) + cot_j;

end

%% =========================================================
%% SPARSE LAPLACIAN MATRIX
%%
%% LapH = L*h
%% =========================================================

L = sparse(Nv,Nv);

for i = 1:Nv

    rows = find(Wcot(i,:));

    wij = Wcot(i,rows);

    L(i,rows) = wij/(2*Ai(i));

    L(i,i) = -sum(wij)/(2*Ai(i));

end

%% =========================================================
%% PRECOMPUTE TRIANGLE GEOMETRY
%% =========================================================

nHat = Nvec ./ ...
    (sqrt(sum(Nvec.^2,2)) + 1e-12);

e1 = cross(nHat, Xi-Xk, 2);

e2 = cross(nHat, Xj-Xi, 2);

%% =========================================================
%% PRECOMPUTE AREA ACCUMULATION
%% =========================================================

area_accum = ...
    accumarray(tri_i,At/3,[Nv 1]) ...
  + accumarray(tri_j,At/3,[Nv 1]) ...
  + accumarray(tri_k,At/3,[Nv 1]);

%% =========================================================
%% INITIAL CONDITION
%% =========================================================

h = 0.05*randn(Nv,1);

%% =========================================================
%% STORAGE
%% =========================================================

time = (0:Nt-1)*dt;

Wrough = zeros(Nt,1);

%% =========================================================
%% OUTPUT DIRECTORY
%% =========================================================

outdir = 'frames';

if ~exist(outdir,'dir')
    mkdir(outdir);
end

%% =========================================================
%% MAIN LOOP
%% =========================================================

for it = 1:Nt

    %% -----------------------------------------------------
    %% PROGRESS
    %% -----------------------------------------------------

    if mod(it,1000)==0

        fprintf('%d / %d\n',it,Nt);

    end

    %% =====================================================
    %% LAPLACE-BELTRAMI
    %% =====================================================

    LapH = L*h;

    %% =====================================================
    %% VECTORIZED FEM GRADIENT
    %% =====================================================

    fi = h(tri_i);
    fj = h(tri_j);
    fk = h(tri_k);

    coef1 = (fj-fi) ./ (2*At);

    coef2 = (fk-fi) ./ (2*At);

    gradf = ...
        coef1 .* e1 ...
      + coef2 .* e2;

    g2 = sum(gradf.^2,2);

    %% =====================================================
    %% VERTEX AVERAGED |grad h|^2
    %% =====================================================

    vals = g2 .* At/3;

    grad2_vertex = ...
        accumarray(tri_i,vals,[Nv 1]) ...
      + accumarray(tri_j,vals,[Nv 1]) ...
      + accumarray(tri_k,vals,[Nv 1]);

    grad2_vertex = ...
        grad2_vertex ./ (area_accum + 1e-12);

    %% =====================================================
    %% STOCHASTIC NOISE
    %% =====================================================

    eta = ...
        (noiseAmp ./ sqrt(Ai*dt)) ...
        .* randn(Nv,1);

    %% =====================================================
    %% EXPLICIT EULER UPDATE
    %% =====================================================

    h = h + dt*(...
        nu*LapH ...
        + 0.5*lambda*grad2_vertex ...
        - kappa*h ...
        + eta);

    %% =====================================================
    %% ROUGHNESS
    %% =====================================================

    hbar = sum(h .* Ai) / sum(Ai);

    Wrough(it) = sqrt(...
        sum(Ai .* (h-hbar).^2) / sum(Ai));

    %% =====================================================
    %% VISUALIZATION
    %% =====================================================

    % if mod(it,50)==0
    % 
    %     X = R0 * V(:,1);
    %     Y = R0 * V(:,2);
    %     Z = R0 * V(:,3);
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
    %     view(3)
    % 
    %     clim([-5 5])
    % 
    %     cb = colorbar;
    % 
    %     cb.Label.String = '$h(\theta,\phi)$';
    % 
    %     cb.Label.Interpreter = 'latex';
    % 
    %     cb.Label.FontSize = 24;
    % 
    %     camlight
    %     lighting gouraud
    % 
    %     xticks([])
    %     yticks([])
    %     zticks([])
    % 
    %     xlabel('')
    %     ylabel('')
    %     zlabel('')
    % 
    %     set(gca,...
    %         'FontSize',18,...
    %         'LineWidth',2)
    % 
    %     drawnow
    % 
    %     % -------------------------------------------------
    %     % SAVE PNG
    %     % -------------------------------------------------
    % 
    %     fname = sprintf('%s/frame_%06d.png',outdir,it);
    % 
    %     exportgraphics(gcf,...
    %         fname,...
    %         'Resolution',100);
    % 
    % end

end

%% =========================================================
%% PLOT W(t)
%% =========================================================

figure('Color','w')

loglog(time,...
    Wrough,...
    '-r',...
    'LineWidth',4)

hold on

loglog(time,...
    3*time.^(1/3),...
    ':k',...
    'LineWidth',3)

xlabel('t')

ylabel('W(t)')

axis square

set(gca,...
    'FontSize',24,...
    'LineWidth',2)

%% =========================================================
%% FUNCTION : COTANGENT
%% =========================================================

function c = Cotangent(a,b)

c = dot(a,b) / ...
   (norm(cross(a,b)) + 1e-12);

end