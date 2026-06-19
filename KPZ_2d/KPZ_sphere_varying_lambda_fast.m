clear; clc; close all;

%% =========================================================
%% FAST KPZ ON TRIANGULATED SPHERE
%%
%% VARYING LAMBDA
%%
%% Explicit Euler
%% Fully vectorized
%%
%% =========================================================

%% =========================================================
%% PARAMETERS
%% =========================================================

R0 = 1;

nu       = 0.1;
kappa    = 0.0;

noiseAmp = 0.05;

Nt = 1000000;

dt = 1e-5;

%% =========================================================
%% LAMBDA VALUES
%% =========================================================

lambda_list = [0 0.2 0.4 0.6 0.8 1.0];

Nlambda = length(lambda_list);

%% =========================================================
%% SPHERE MESH
%% =========================================================

Npts = 128;

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
%% TRIANGLE GEOMETRY
%% =========================================================

Xi = V(tri_i,:);
Xj = V(tri_j,:);
Xk = V(tri_k,:);

Nvec = cross(Xj-Xi,Xk-Xi,2);

At = 0.5*sqrt(sum(Nvec.^2,2));

%% =========================================================
%% BARYCENTRIC AREA
%% =========================================================

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
%% SPARSE LAPLACE-BELTRAMI MATRIX
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
%% AREA ACCUMULATION
%% =========================================================

area_accum = ...
    accumarray(tri_i,At/3,[Nv 1]) ...
  + accumarray(tri_j,At/3,[Nv 1]) ...
  + accumarray(tri_k,At/3,[Nv 1]);

%% =========================================================
%% STORAGE
%% =========================================================

time = (0:Nt-1)*dt;

W_all = zeros(Nt,Nlambda);

%% =========================================================
%% LOOP OVER LAMBDA
%% =========================================================

for ilam = 1:Nlambda

    lambda = lambda_list(ilam);

    fprintf('\nRunning lambda = %.3f\n',lambda);

    %% =====================================================
    %% INITIAL CONDITION
    %% =====================================================

    h = 0.05*randn(Nv,1);

    Wrough = zeros(Nt,1);

    %% =====================================================
    %% MAIN TIME LOOP
    %% =====================================================

    for it = 1:Nt

        %% -------------------------------------------------
        %% progress
        %% -------------------------------------------------

        if mod(it,5000)==0

            fprintf('%d / %d\n',it,Nt);

        end

        %% =================================================
        %% LAPLACE-BELTRAMI
        %% =================================================

        LapH = L*h;

        %% =================================================
        %% FEM SURFACE GRADIENT
        %% =================================================

        fi = h(tri_i);
        fj = h(tri_j);
        fk = h(tri_k);

        coef1 = (fj-fi) ./ (2*At);

        coef2 = (fk-fi) ./ (2*At);

        gradf = ...
            coef1 .* e1 ...
          + coef2 .* e2;

        g2 = sum(gradf.^2,2);

        %% =================================================
        %% VERTEX-AVERAGED |grad h|^2
        %% =================================================

        vals = g2 .* At/3;

        grad2_vertex = ...
            accumarray(tri_i,vals,[Nv 1]) ...
          + accumarray(tri_j,vals,[Nv 1]) ...
          + accumarray(tri_k,vals,[Nv 1]);

        grad2_vertex = ...
            grad2_vertex ./ (area_accum + 1e-12);

        %% =================================================
        %% STOCHASTIC NOISE
        %% =================================================

        eta = ...
            (noiseAmp ./ sqrt(Ai*dt)) ...
            .* randn(Nv,1);

        %% =================================================
        %% EXPLICIT EULER UPDATE
        %% =================================================

        h = h + dt*(...
            nu*LapH ...
            + 0.5*lambda*grad2_vertex ...
            - kappa*h ...
            + eta);

        %% =================================================
        %% ROUGHNESS
        %% =================================================

        hbar = sum(h .* Ai) / sum(Ai);

        Wrough(it) = sqrt(...
            sum(Ai .* (h-hbar).^2) / sum(Ai));

    end

    %% =====================================================
    %% STORE
    %% =====================================================

    W_all(:,ilam) = Wrough;

end

%% =========================================================
%% PLOT
%% =========================================================

% colors = [
%     0.2 0.2 0.2
%     0.2 0.4 0.9
%     0.2 0.7 0.3
%     0.9 0.3 0.2
% ];

figure('Color','w',...
    'Position',[100 100 800 700])

for ilam = 1:Nlambda

    loglog(time,...
        W_all(:,ilam),...
        'LineWidth',4,...
        'DisplayName',...
        ['\lambda = ',num2str(lambda_list(ilam))]);

    hold on

end

hold on

loglog(time,...
    2*time.^(1/3),...
    ':k',...
    'LineWidth',3,...
    'DisplayName','t^{1/3}')

xlabel('t')

ylabel('W(t)')

legend('Location','northwest',...
    'FontSize',20)

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