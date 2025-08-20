%% 初始化华数机器人模型
close all;clear;
global ToRad ToDeg hs;
ToRad=pi/180.0;
ToDeg=180.0/pi;  
load('tool.mat');
% load('tar.mat');
% ta=tar;
%% 工具箱建模
L1 = Link( 'd', 0, 'a', 0, 'alpha', -1*pi/2,'qlim',[-180 180]*ToRad);
L2 = Link( 'd', -126.5/1000.0, 'a', 726/1000.0, 'alpha', 0*pi/2,'qlim',[-85 85]*ToRad,'offset',-pi/2);%注意，offset相当于旋转了z轴，会影响到后面的坐标系
L3 = Link( 'd', -96/1000.0, 'a', 0, 'alpha', pi/2,'qlim',[-40 220]*ToRad,'offset',-pi);
L4 = Link( 'd', 630.5/1000.0, 'a', 0, 'alpha',pi/2,'qlim',[-90 90]*ToRad);
L5 = Link( 'd', 91.0/1000.0, 'a', 0, 'alpha',- pi/2,'qlim',[-90 90]*ToRad,'offset',0);
L6 = Link( 'd', 122.0/1000.0, 'a', 0, 'alpha', 0*pi/2,'qlim',[-180 180]*ToRad);
hs=SerialLink([L1,L2,L3,L4,L5,L6],'base',transl(0,0,420.5/1000.0),'tool',torch_end);%SerialLink 类函数
hs.name='HSR';
theta=[0,0,0,0,0,0]*ToRad;
W = [-1.5 1.5,-1.5,1.5,-1,2.0];
% hs.plot(theta,'workspace',W,'view',[142,22]);
% hold on;
% plottcp(se2t(hs.fkine(theta)));
% se2t(hs.fkine(theta))
%%
% hs.teach();
% plottcp(tar);
% a = se2t(hs.fkine(theta));
% b = tar;n=10;
% % tc = ctraj(a, b,n);
% % for i=2:n-1
% %     plottcp(tc(:,:,i));
% % end
% % 逆解达点自由空间不考虑障碍
% q = theta;
% alfa = 0.1;
% e = zeros(6,1);
% while true
%     Tq = hs.fkine(q);
% %     tc =  ctraj(se2t(Tq), tar,5);
%     e = tr2delta(Tq,tar);
%     J = hs.jacobe(q);
%     dq = alfa*pinv(J)*e;
% %     dq.*180/pi
%     q(1)=q(1)+dq(1);
%     if q(1)<hs.qlim(1,1)
%         q(1)=hs.qlim(1,1);
%     end
%     if q(1)>hs.qlim(1,2)
%         q(1)=hs.qlim(1,2);
%     end
%     q(2)=q(2)+dq(2);
%     if q(2)<hs.qlim(2,1)
%         q(2)=hs.qlim(2,1);
%     end
%     if q(2)>hs.qlim(2,2)
%         q(2)=hs.qlim(2,2);
%     end
%     q(3)=q(3)+dq(3);
%     if q(3)<hs.qlim(3,1)
%         q(3)=hs.qlim(3,1);
%     end
%     if q(3)>hs.qlim(3,2)
%         q(3)=hs.qlim(3,2);
%     end
%     q(4)=q(4)+dq(4);
%     if q(4)<hs.qlim(4,1)
%         q(4)=hs.qlim(4,1);
%     end
%     if q(4)>hs.qlim(4,2)
%         q(4)=hs.qlim(4,2);
%     end
%     q(5)=q(5)+dq(5);
%     if q(5)<hs.qlim(5,1)
%         q(5)=hs.qlim(5,1);
%     end
%     if q(5)>hs.qlim(5,2)
%         q(5)=hs.qlim(5,2);
%     end
%     q(6)=q(6)+dq(6);
%     if q(6)<hs.qlim(6,1)
%         q(6)=hs.qlim(6,1);
%     end
%     if q(6)>hs.qlim(6,2)
%         q(6)=hs.qlim(6,2);
%     end
%     hs.plot(q);
%     plottcp(se2t(hs.fkine(q)));
%     if norm(e) < 1e-2
%         q=q./pi*180.0;
%         break; 
%     end
%     
% end
