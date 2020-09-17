function [ Zstar, Z ] = getAnchorZnormalized( X )
    viewNum = size(X,2);   
    N = size(X{1},1);      
    if N > 5000
        p = 1000;
    else
        p = ceil(N/10);
    end
    allIndex = randperm(N);
    anchorIndex = allIndex(1:p);
    for it = 1:viewNum
        Anchor{it} = X{it}(anchorIndex,:);
    end
    fprintf('Nonlinear Anchor Embedding...\n');
    for it = 1:viewNum
        fprintf('The %d-th view Nonlinear Anchor Embeeding...\n',it);
        dist = EuDist2(X{it},Anchor{it},0);
        sigma = mean(min(dist,[],2).^0.5)*2;
        feaVec = exp(-dist/(2*sigma*sigma));
        Z{it} = bsxfun(@minus, feaVec', mean(feaVec',2));
    end
    
    Zstar = zeros(size(Z{1}));
    for i = 1:viewNum
        Zstar = Zstar + Z{i};
    end
    sumZstar = sum(Zstar);
    Zstar = Zstar./sum(Zstar);
    D = ones(1,N)*Zstar'*Zstar;
    Zstar = Zstar./sqrt(abs(D));
    Zstar = Zstar';
end

