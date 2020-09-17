function [result] = SGMVC(Z,Zstar,Y)
result = [];
%--------------------Parameters--------------------------------------------
MaxIter =  15;           % 7 iterations are okay, but better results for 15
innerMax = 5;            % maximum iterations for subproblem of B and C
L = 128;                 % Hashing code length
viewNum = size(Z,2);     % number of views
N = size(Z{1},2);
beta = viewNum/L;        % Hyper-para beta
gamma = 1/N;
lambda = 1/N;
%--------------------End Parameters----------------------------------------


%--------------------Initialization----------------------------------------
viewNum = size(Z,2); % number of views
n_cluster = numel(unique(Y));
alpha = ones(viewNum,1) / N;
alpha_r = alpha;

if length(Z)<4
    sel_sample = Z{1}(:,randsample(N, 1000),:);
    [pcaW, ~] = eigs(cov(sel_sample'), L);
    B = sign(pcaW'*Z{1});
else
    sel_sample = Z{4}(:,randsample(N, 1000),:);
    [pcaW, ~] = eigs(cov(sel_sample'), L);
    B = sign(pcaW'*Z{4});
end

P = cell(1,viewNum);
C = B(:,randsample(N, n_cluster));
HamDist = 0.5*(L - B'*C);
[~,ind] = min(HamDist,[],2);
G = sparse(ind,1:N,1,n_cluster,N,N);
G = full(G);
CG = C*G;

ZZT = cell(1,viewNum);
for view = 1:viewNum
    ZZT{view} = Z{view}*Z{view}';
end
clear HamDist ind initInd n_randm pcaW sel_sample view
%--------------------End Initialization------------------------------------


%--------------------The proposed method-----------------------------------
disp('----------The proposed method (multi-view)----------');
diagA = ones(1,N)*Zstar*Zstar';

for iter = 1:MaxIter
    fprintf('The %d-th iteration...\n',iter);
    %---------Update Pv--------------
    PX = zeros(L,N);
    for v = 1:viewNum
        P{v} = B*Z{v}'/(ZZT{v}+beta*eye(size(Z{v},1)));
        PX   = PX+alpha_r(v)*P{v}*Z{v};
    end
    
    %---------Update B--------------
     for iterIn = 1:3
        muB = 1e-4;
        BdiagA = B.* diagA;
        BZ1 = B*Zstar*Zstar';
        gradientB = -2*(PX+lambda*CG) + 2*gamma*(BdiagA-BZ1);
        B         = sign(B-1/muB*gradientB);
        B(B==0)   = 1;
    end
 
    %---------Update C and G--------------
    for iterInner = 1:innerMax
        C = sign(B*G'); C(C==0) = 1;
        rho = .001; mu = .01; % Preferred for this dataset
        for iterIn = 1:3
            grad = -B*G' + rho*repmat(sum(C),L,1);
            C    = sign(C-1/mu*grad); C(C==0) = 1;
        end
        
        HamDist = 0.5*(L - B'*C); 
        [~,indx] = min(HamDist,[],2);
        G = sparse(indx,1:N,1,n_cluster,N,N);
    end
    CG = C*G;
  
    %---------Update alpha--------------
    h = zeros(viewNum,1);
    for view = 1:viewNum
        h(view) = 1/(2*norm(B-P{view}*Z{view},'fro'));
    end
    alpha_r = h;
    
   [~,pred_label] = max(G,[],1);
   res_cluster = ClusteringMeasure(Y, pred_label);
   fprintf('All view results: ACC = %.4f and NMI = %.4f, Purity = %.4f\n\n',res_cluster(1),res_cluster(2),res_cluster(3));
   result = [result; res_cluster];
end
disp('----------Main Iteration Completed----------');
end