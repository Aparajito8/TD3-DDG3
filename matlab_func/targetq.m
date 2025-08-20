function dq = targetq(q,T)
global hs;
alfa = 0.8;
Tq = hs.fkine(q);
e = tr2delta(Tq,T);
J = hs.jacobe(q);
dq = alfa*pinv(J)*e;
% dq = J'*inv(J*J'+alfa^2.*eye(6))*e;

end

