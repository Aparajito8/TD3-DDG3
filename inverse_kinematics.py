#
#
# 用于测试 matlab 逆运动学 和 通信 以及运动小球的控制


import numpy as np
import vrep
import time
import matlab.engine
import matlab


class robot():
    eng = matlab.engine.connect_matlab('robot')
    dt = 0.05
    ToRad = np.pi / 180.0
    jointNum = 6
    jointName = 'j'
    TipName = 'Tip'
    TargetpName = 'target#0'
    np.random.seed(6)
    def __init__(self):
        # self.eng.eval('cd E:\ 02_td3_DDPG\matlab_func',nargout=0)
        self.eng.eval('init',nargout = 0)

        # 建立通讯
        vrep.simxFinish(-1)
        # 每隔0.2s检测一次，直到连接上V-rep
        while True:
            self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
            if self.clientID != -1:
                break
            else:
                time.sleep(0.2)
                print("Failed connecting to remote API server!")
        print("Connection success!")
        # 设置机械臂仿真步长，
        vrep.simxSetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, self.dt,vrep.simx_opmode_oneshot)
        # 打开同步模式
        vrep.simxSynchronous(self.clientID, True)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)

        # 获取句柄 joint
        self.jointHandle = np.zeros((self.jointNum,), dtype=np.int)
        for i in range(self.jointNum):
            _, returnHandle = vrep.simxGetObjectHandle(self.clientID, self.jointName + str(i + 1),vrep.simx_opmode_blocking)
            self.jointHandle[i] = returnHandle
        # 移动障碍的平移轴句柄
        _, self.movHandle = vrep.simxGetObjectHandle(self.clientID, 'mov',vrep.simx_opmode_blocking)

        _, self.TipHandle = vrep.simxGetObjectHandle(self.clientID, self.TipName, vrep.simx_opmode_blocking)
        _, self.TargetHandle = vrep.simxGetObjectHandle(self.clientID, self.TargetpName, vrep.simx_opmode_blocking)
        _, self.refframeHandle = vrep.simxGetObjectHandle(self.clientID, 'base_ref', vrep.simx_opmode_blocking)


        _, tarPos = vrep.simxGetObjectPosition(self.clientID,self.TargetHandle,self.refframeHandle,vrep.simx_opmode_oneshot_wait)
        _, tarRot = vrep.simxGetObjectOrientation(self.clientID, self.TargetHandle, self.refframeHandle,vrep.simx_opmode_oneshot_wait)
        self.Tar = self.vrep2Tr(tarPos,tarRot)

        self.jointpos_now = np.zeros((self.jointNum,), np.float)
        for i in range(self.jointNum):
            _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_streaming)
        _, self.mov_pos_now = vrep.simxGetJointPosition(self.clientID, self.movHandle,vrep.simx_opmode_streaming)

    def inverse(self):
        cnt =0
        cnt_max = 200
        idx = np.arange(1,cnt_max+1)
        turn_idx = np.sort(np.random.choice(idx,15,False))
        turn_idx = np.array([30,140,180])
        np.save('turn_idx.npy', turn_idx)
        numpy_array = np.load('turn_idx.npy')
        v = 0.3

        while True:

            cnt+=1
            print(cnt)
            if (cnt in turn_idx) == True :
                v=-v

            _, self.mov_pos_now = vrep.simxGetJointPosition(self.clientID, self.movHandle,vrep.simx_opmode_buffer)


            mov_new = v*self.dt+self.mov_pos_now
            print(mov_new)

            _, tarPos = vrep.simxGetObjectPosition(self.clientID, self.TargetHandle, self.refframeHandle,
                                                   vrep.simx_opmode_oneshot_wait)
            _, tarRot = vrep.simxGetObjectOrientation(self.clientID, self.TargetHandle, self.refframeHandle,
                                                      vrep.simx_opmode_oneshot_wait)
            self.Tar = self.vrep2Tr(tarPos, tarRot)
            for i in range(self.jointNum):
                _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_buffer)
            tar_q = self.jointpos_now
            dq = self.eng.targetq(matlab.double(self.jointpos_now.data),matlab.double(self.Tar.tolist()), nargout=1)
            dq = np.squeeze(np.array(dq))
            next_q = dq*self.dt+tar_q
            vrep.simxPauseCommunication(self.clientID, True)
            for i in range(self.jointNum):  # 关节归位
                vrep.simxSetJointTargetPosition(self.clientID, self.jointHandle[i], next_q[i],vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.movHandle, mov_new, vrep.simx_opmode_oneshot)
            vrep.simxPauseCommunication(self.clientID, False)
            vrep.simxSynchronousTrigger(self.clientID)
            vrep.simxGetPingTime(self.clientID)
            if np.linalg.norm(dq)<1e-2 or cnt==cnt_max:
                vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
                break

    def vrep2Tr(self,pos,rot):
        R = self.eul2RotM(rot)
        t = np.asarray(pos)
        t = t[:,np.newaxis]
        Tr = np.column_stack((R,t))
        a = np.array([[0,0,0,1]])
        Tr = np.row_stack((Tr,a))
        return Tr

    def eul2RotM(self,rpy):
        Rx = np.array([[1,0,0],
                       [0,np.cos(rpy[0]),-np.sin(rpy[0])],
                       [0,np.sin(rpy[0]),np.cos(rpy[0])]])
        Ry = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])],
                       [0, 1, 0],
                       [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])
        Rz = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]),0],
                       [np.sin(rpy[2]), np.cos(rpy[2]),0],
                       [0,0,1]])
        Rot = np.matmul(Rz,np.matmul(Ry,Rx))
        return Rot

if  __name__ == '__main__':
    robot = robot()
    robot.inverse()
    print('end!')