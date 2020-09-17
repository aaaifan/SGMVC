clear all;

dataset = 'DIGIT.mat';
load(dataset)
[Zstar, Z] = getAnchorZnormalized(X);
[result] = SGMVC(Z, Zstar ,Y);
fprintf('acc=%.4f,nmi=%.4f,purity=%.4f\n',result(end,1),result(end,2),result(end,3));

