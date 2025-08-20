clear;ToRad=pi/180.0;
init;
robot = importrobot('URDFV3.urdf');
robot.DataFormat = 'row';
W = [-1 1,-0.6,0.6,0,1.2];

figure(1);
axis(W);
% axis off
robot.Bodies{1, 1}.Joint.PositionLimits= [-180 180]*ToRad;
robot.Bodies{1, 2}.Joint.PositionLimits= [-85 85]*ToRad;
robot.Bodies{1, 3}.Joint.PositionLimits= [-40 220]*ToRad;
robot.Bodies{1, 4}.Joint.PositionLimits= [-90 90]*ToRad;
robot.Bodies{1, 5}.Joint.PositionLimits= [-90 90]*ToRad;
robot.Bodies{1, 6}.Joint.PositionLimits= [-180 180]*ToRad;
show(robot,[0,0,0,0,0,0]*ToRad,'Visuals','on','Frames','off');
axis(W);
view(118,16);
figure(2);
trplot(transl(0,0,0),'length',0.2,'rgb','thick',2,'notext','axis',W,'framelabel','0');hold on;
trplot(transl(0,0,0.4205),'length',0.2,'rgb','thick',2,'notext','axis',W,'framelabel','1');
for i=1:5
    if i == 1
        T0 = transl(0,0,0.4205)*se2t(hs.A(i,theta));
        trplot(T0,'length',0.2,'rgb','thick',2,'notext','axis',W,'framelabel','2');
    end
    if i==2||i==3
        T = transl(0,0,0.4205)*se2t(hs.A(1:i,theta));
        trplot(T,'length',0.1,'rgb','thick',2,'notext','axis',W);
    end
    T = transl(0,0,0.4205)*se2t(hs.A(1:i,theta));
        trplot(T,'length',0.05,'rgb','thick',2,'notext','axis',W);
end
view(118,16);
