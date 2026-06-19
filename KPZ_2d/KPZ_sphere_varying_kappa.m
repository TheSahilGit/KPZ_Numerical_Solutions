clear; clc; close all;

%% =========================================================
%% KPZ ON TRIANGULATED SPHERE
%%
%% VARY kappa
%%
%% kappa = 1e-2 -> 1e2
%%
%% Everything else unchanged
%%
%% =========================================================

%% =========================================================
%% PARAMETERS
%% =========================================================

R0 = 1;

nu       = 0.1;
lambda   = 0.05;

noiseAmp = 1.0;

vo = 0.0; 

Nt = 10000;

dt = 1e-3;

%% =========================================================
%% KAPPA VALUES
%% =========================================================

kappa_list = [1e-2 1e-1 1e0 1e1 1e2];

Nkappa = length(kappa_list);

%% =========================================================
%% SPHERE MESH : FIBONACCI SPHERE + CONVEX HULL
%% =========================================================

Npts = 1024;

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

%% =========================================================
%% PRECOMPUTE COTANGENT WEIGHTS
%% =========================================================

Wcot = sparse(Nv,Nv);

for t = 1:Ntmesh

    ids = tri(t,:);

    i = ids(1);
    j = ids(2);
    k = ids(3);

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
%% STORAGE
%% =========================================================

time = (0:Nt-1)*dt;

W_all = zeros(Nt,Nkappa);

%% =========================================================
%% LOOP OVER kappa
%% =========================================================

for ik = 1:Nkappa

    kappa = kappa_list(ik);

    disp(['Running kappa = ', num2str(kappa)])

    %% =====================================================
    %% INITIAL CONDITION
    %% =====================================================

    h = 0.05*randn(Nv,1);

    Wrough = zeros(Nt,1);

    %% =====================================================
    %% TIME LOOP
    %% =====================================================

    for it = 1:Nt

        %disp(it)

        %% =================================================
        %% LAPLACE-BELTRAMI
        %% =================================================

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

        end

        %% =================================================
        %% GRADIENT TERM
        %% =================================================

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

            Nvec = cross(xj-xi,xk-xi);

            At = 0.5*norm(Nvec);

            nHat = Nvec/(norm(Nvec)+1e-12);

            e1 = cross(nHat,xi-xk);
            e2 = cross(nHat,xj-xi);

            gradf = ...
                (fj-fi)*e1/(2*At) ...
              + (fk-fi)*e2/(2*At);

            g2 = dot(gradf,gradf);

            grad2_vertex(i) = grad2_vertex(i) + g2*At/3;
            grad2_vertex(j) = grad2_vertex(j) + g2*At/3;
            grad2_vertex(k) = grad2_vertex(k) + g2*At/3;

            area_accum(i) = area_accum(i) + At/3;
            area_accum(j) = area_accum(j) + At/3;
            area_accum(k) = area_accum(k) + At/3;

        end

        grad2_vertex = ...
            grad2_vertex ./ (area_accum + 1e-12);

        %% =================================================
        %% STOCHASTIC NOISE
        %% =================================================

        eta = ...
            (noiseAmp./sqrt(Ai*dt)) ...
            .* randn(Nv,1);

        %% =================================================
        %% EULER UPDATE
        %% =================================================

        h_new = h + dt*(...
            nu*LapH ...
            + 0.5*lambda*grad2_vertex ...
            - kappa*h ...
            + eta + vo);

        h = h_new;

        %% =================================================
        %% ROUGHNESS
        %% =================================================
        % 
        % R = R0 + h;
        % 
        % Rbar = sum(R .* Ai) / sum(Ai);
        % 
        % Wrough(it) = sqrt(...
        %     sum(Ai .* (R-Rbar).^2) / sum(Ai));


    
        hbar = sum(h .* Ai) / sum(Ai);
    
        Wrough(it) = sqrt(...
         sum(Ai .* (h-hbar).^2) / sum(Ai));


    end

    %% =====================================================
    %% STORE
    %% =====================================================

    W_all(:,ik) = Wrough;

end

%% =========================================================
%% PLOT
%% =========================================================

figure('Color','w', 'Position',[200 200 600 600])

colors = [
    0.3 0.9 0.9
    0.2 0.4 0.9
    0.9 0.3 0.2
    0.2 0.7 0.3
    0.6 0.2 0.7
];

%time = time./max(time);

for ik = 1:Nkappa

    loglog(time,...
        W_all(:,ik),...
        'Color',colors(ik,:),...
        'LineWidth',6,...
        'DisplayName',...
        ['\kappa = ',num2str(kappa_list(ik))]);
        hold on

end

loglog(time(1:100),...
    4.2*time(1:100).^(1/3),...
    ':k',...
    'LineWidth',4,...
    'DisplayName','t^{1/3}')

hold on
loglog(time(101:end),...
    2*time(101:end).^(1/10),...
    '-.k',...
    'LineWidth',4,...
    'DisplayName','t^{1/10}')

xlabel('t')

ylabel('W(t)')

legend('Location','northwest',...
    'FontSize',20)

axis square

set(gca,...
    'FontSize',22,...
    'LineWidth',2)


%% =========================================================
%% FUNCTION : COTANGENT
%% =========================================================

function c = Cotangent(a,b)

c = dot(a,b)/(norm(cross(a,b))+1e-12);

end