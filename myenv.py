'''
毕业论文2021 第四章
 hsr 华数动态避障环境

 华数机器人初始角度【0，0，0，0，0，0】

'''

import numpy as np
import vrep
import time
import gym.spaces
import matlab.engine
import matlab
import math
# np.random.seed(6)
class hsr():
    '''
    :param 定义常量
    '''
    ToRad = np.pi / 180.0
    jointNum = 6
    jointName = 'j'
    TipName = 'Tip'
    TargetName = 'target#0'
    eng = matlab.engine.connect_matlab('robot')
    dt = 0.05
    T = 250
    action_bound = [-np.pi*2,np.pi*2]
    dq_bound = np.array([[-45., 45.],
                         [-37., 37.],
                         [-45., 45.],
                         [-56., 56.],
                         [-56., 56.],
                         [-60., 60.]]) * ToRad * dt * 2
    action_dim = 6
    initialConfig = np.array([0.,0.,0.,0.,0.,0.]) * ToRad
    init_mov = 0.2
    action_space = gym.spaces.Box(low=action_bound[0], high=action_bound[1], shape=[action_dim])
    turn_idx = np.array([20, 135, 180])
    v_mov = 0.35
    delta = 0.05
    def __init__(self):
        '''
        :param 初始化环境，与vrep握手通信
        '''
        self.eng.eval('init', nargout=0)  # 初始化数据
        self.state_dim = 11
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
        # 设置机械臂步长
        vrep.simxSetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, self.dt,vrep.simx_opmode_oneshot)
        #打开同步模式
        vrep.simxSynchronous(self.clientID, True)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        # 获取句柄
        # 关节角度句柄
        self.jointHandle = np.zeros((self.jointNum,), dtype=np.int)
        for i in range(self.jointNum):
            _,returnHandle=vrep.simxGetObjectHandle(self.clientID,self.jointName+str(i+1),vrep.simx_opmode_blocking)
            self.jointHandle[i]=returnHandle
        # 运动小球的移动轴句柄
        _, self.movHandle = vrep.simxGetObjectHandle(self.clientID, 'mov', vrep.simx_opmode_blocking)

        _, self.TipHandle = vrep.simxGetObjectHandle(self.clientID, self.TipName, vrep.simx_opmode_blocking)
        _, self.TargetHandle = vrep.simxGetObjectHandle(self.clientID, self.TargetName, vrep.simx_opmode_blocking)
        _, self.refframeHandle = vrep.simxGetObjectHandle(self.clientID, 'base_ref', vrep.simx_opmode_blocking)

        _, tarPos = vrep.simxGetObjectPosition(self.clientID, self.TargetHandle, self.refframeHandle,vrep.simx_opmode_oneshot_wait)
        _, tarRot = vrep.simxGetObjectOrientation(self.clientID, self.TargetHandle, self.refframeHandle,vrep.simx_opmode_oneshot_wait)
        self.Tar = self.vrep2Tr(tarPos, tarRot)

        # 获取计算句柄 距离检测
        _,self.obsdisHandle = vrep.simxGetDistanceHandle(self.clientID,'obsdis',vrep.simx_opmode_blocking)
        _,self.tardisHandle = vrep.simxGetDistanceHandle(self.clientID,'tardis',vrep.simx_opmode_blocking)

        # 初始化数据流
        res,self.tipPosNow = vrep.simxGetObjectPosition(self.clientID, self.TipHandle, self.refframeHandle,vrep.simx_opmode_oneshot_wait)#末端实时位置
        print('tippos', self.tipPosNow)

        self.jointpos_now = np.zeros((self.jointNum,), np.float)
        for i in range(self.jointNum):
            _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_streaming)
        print('joint', self.jointpos_now)

        _, self.movepos_now= vrep.simxGetJointPosition(self.clientID, self.movHandle,vrep.simx_opmode_streaming)
        print('moveinit',self.movepos_now)

        res,self.obsdist = vrep.simxReadDistance(self.clientID, self.obsdisHandle, vrep.simx_opmode_streaming)
        res,self.tardist = vrep.simxReadDistance(self.clientID, self.tardisHandle, vrep.simx_opmode_streaming)
        print('obsdist',self.obsdist)
        print('tardist', self.obsdist)
        print('Handles available!')

    def reset(self):
        # 关节归位
        vrep.simxSynchronousTrigger(self.clientID)  # 让仿真走一步
        vrep.simxGetPingTime(self.clientID)
        vrep.simxPauseCommunication(self.clientID, True)
        for i in range(self.jointNum):  # 关节归位
            vrep.simxSetJointTargetPosition(self.clientID, self.jointHandle[i], self.initialConfig[i],vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(self.clientID,self.movHandle,self.init_mov,vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, False)
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        self.jointpos_now = np.zeros((self.jointNum,), np.float)
        for i in range(self.jointNum):
            _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_buffer)
        _,self.movepos_now = vrep.simxGetJointPosition(self.clientID, self.movHandle,vrep.simx_opmode_buffer)
        while abs(self.jointpos_now[0] - self.initialConfig[0]) > 1e-2 or abs(
                self.jointpos_now[1] - self.initialConfig[1]) > 1e-2 or abs(
                self.jointpos_now[2] - self.initialConfig[2]) > 1e-2 \
                or abs(self.jointpos_now[3] - self.initialConfig[3]) > 1e-2 or abs(
            self.jointpos_now[4] - self.initialConfig[4]) > 1e-2 or abs(
            self.jointpos_now[5] - self.initialConfig[5]) > 1e-2 or abs(self.movepos_now-self.init_mov)>1e-2:
            vrep.simxSynchronousTrigger(self.clientID)
            vrep.simxGetPingTime(self.clientID)
            for i in range(self.jointNum):
                _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_buffer)
            _, self.movepos_now = vrep.simxGetJointPosition(self.clientID, self.movHandle, vrep.simx_opmode_buffer)
        self.done = False
        self.t = 0
        self.v = self.v_mov
        # 返回当前环境中的状态向量 【s_0】
        for i in range(self.jointNum):
            _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_buffer)
        _,tipPosNow =  vrep.simxGetObjectPosition(self.clientID, self.TipHandle, self.refframeHandle,vrep.simx_opmode_oneshot_wait)
        self.tipPosNow = np.array(tipPosNow)
        _,self.dtar = vrep.simxReadDistance(self.clientID,self.tardisHandle,vrep.simx_opmode_buffer)
        _,self.do = vrep.simxReadDistance(self.clientID,self.obsdisHandle,vrep.simx_opmode_buffer)
        s = self.jointpos_now
        s = np.hstack((s,self.tipPosNow))
        s = np.hstack((s,np.array([self.do])))
        s = np.hstack((s,np.array([self.dtar])))
        dq = self.eng.targetq(matlab.double(self.jointpos_now.data), matlab.double(self.Tar.tolist()), nargout=1)
        self.dq = np.clip(np.array(dq).reshape((-1,)), *self.action_bound)
        return s

    def step(self,action):
        '''

        :param action: 输入的六维动作，numpy 6维向量
        :return:
        '''
        self.t+=1
        if (self.t in self.turn_idx) == True:
            self.v = -self.v
        self.done = self.t==self.T
        for i in range(self.jointNum):
            _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_buffer)
        _, self.movepos_now = vrep.simxGetJointPosition(self.clientID, self.movHandle, vrep.simx_opmode_buffer)
        self.action_new = action*0.1+ self.dq
        targetJointPos = self.jointpos_now + self.limV(self.action_new*self.dt)
        targetMovPos = self.movepos_now +self.v*self.dt
        vrep.simxPauseCommunication(self.clientID, True)
        for i in range(self.jointNum):  # 关节归位
            vrep.simxSetJointTargetPosition(self.clientID, self.jointHandle[i], targetJointPos[i],vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetPosition(self.clientID,self.movHandle,targetMovPos,vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, False)
        vrep.simxSynchronousTrigger(self.clientID)
        vrep.simxGetPingTime(self.clientID)
        # 获取转移状态
        for i in range(self.jointNum):
            _, self.jointpos_now[i] = vrep.simxGetJointPosition(self.clientID, self.jointHandle[i],vrep.simx_opmode_buffer)
        _,tipPosNow =  vrep.simxGetObjectPosition(self.clientID, self.TipHandle, self.refframeHandle,vrep.simx_opmode_oneshot_wait)
        self.tipPosNow = np.array(tipPosNow)
        _,self.dtar = vrep.simxReadDistance(self.clientID,self.tardisHandle,vrep.simx_opmode_buffer)
        _,self.do = vrep.simxReadDistance(self.clientID,self.obsdisHandle,vrep.simx_opmode_buffer)
        s_ = self.jointpos_now
        s_ = np.hstack((s_,self.tipPosNow))
        s_ = np.hstack((s_,np.array([self.do])))
        s_ = np.hstack((s_,np.array([self.dtar])))
        dq = self.eng.targetq(matlab.double(self.jointpos_now.data), matlab.double(self.Tar.tolist()), nargout=1)
        self.dq = np.clip(np.array(dq).reshape((-1,)), *self.action_bound)
        # 计算奖励
        Rt = -0.5 * self.dtar ** 2 if self.dtar < self.delta else -self.delta * (self.dtar - 0.5 * self.delta)
        r = Rt*50  # d_tar:0-0.5  r:0-1.2
        # Ra = np.linalg.norm(action)#-np.linalg.norm(np.array(tipRot))
        # r-=np.tanh(Ra)*0.1
        if self.do<=0.01:
            r-=2.5
        if self.dtar<=0.05:
            r+=1.5
        return s_,r,self.done
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

    def limV(self,action):
        for i in range(self.jointNum):
            if action[i]<self.dq_bound[i,0]:
                action[i] = self.dq_bound[i,0]
            if action[i]>self.dq_bound[i,1]:
                action[i] = self.dq_bound[i,1]
        return action

    def sampleAction(self):
        a = np.random.normal(size=(self.jointNum,))
        return a

    def stopSim(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        return 0
if  __name__ == '__main__':
    robot = hsr()
    print(robot.reset())
    ep_r = 0
    for i in range(300):
        s_,r,done =robot.step(robot.sampleAction())
        print(r)
        ep_r+=r
        if done:
            break
    robot.stopSim()
    print(ep_r)


