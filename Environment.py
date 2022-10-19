from gym import Env
import numpy as np
from gym.spaces import Discrete, Box
from PID_Simulation import simulationfunction
import random
from collections import deque


class ControlModelEnv(Env):
    def __init__(self):
        self.n_actions = 101
        self.resolution = 0.15
        self.P_parameter=0.003
        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=np.float32(np.array([-10, -10, -10, -10, -10])),
                                     high=np.float32(np.array([10, 10, 10, 10, 10])))
        self.amp_soll = 10
        self.amp_frei = 20
        self.u_new = 0
        self.target_deflection = 0
        self.int_target_error = 0
        self.piezo = [0, 0, 0, 0, 0, 0]
        
        # Laden der Scanlinie 
        self.top = []                    # top = Topographie
        self.line_file = 'line1.dat'
        self.load_line()
        
        self.state = deque([0, 0, 0, 0, 0], maxlen=5)
        # Länge der Punkte die für die Berechnung genutzt werden
        self.scan_lenth = 1000
        self.max_lenth = 1000
        self.top_start = 0

    def reset(self):
        # self.i = random.randint(0, 2000)
        self.i = random.randint(0, 2000) # Zufälliger Anfangspunkt der Scanlinie
        self.start_line = self.i
        self.u_new = 0
        self.target_deflection = 0
        self.piezo = [0, 0, 0, 0, 0, 0]
        self.state = deque([0, 0, 0, 0, 0], maxlen=5)
        self.scan_lenth = 1000
        self.max_lenth = 1000
        self.top_start = 0
        self.amp_soll = 10
        self.amp_frei = 20
        # ft=np.random.random_sample(1);
        # hr=np.random.random_sample(1);
        # self.top = (self.top_orig*ft+(1-ft)*np.cumsum(np.random.randn(3001)))
        return np.array(self.state)

    def step(self, action):
        if self.scan_lenth == self.max_lenth:
            self.top_start = self.top[self.i]
        self.scan_lenth -= 1
        # next state transition 
        # Spannungsberechnung
        delta_deflection=(action - ((self.n_actions + 1)/2)) * self.resolution
        self.target_deflection += delta_deflection
        deflection_error=self.target_deflection-self.piezo[0]
        delta_u = deflection_error*self.P_parameter
        self.u_new += delta_u
        self.piezo = simulationfunction(self.piezo[0], self.piezo[1], self.piezo[2],
                                        self.piezo[3], self.piezo[4], self.piezo[5], self.u_new)
        # Berechnung Schwingungsampl
        amp = self.amp_soll - (self.piezo[0] + self.top[self.i] - self.top_start)
        if amp < 0:
            amp = 0
        if amp > self.amp_frei:
            amp = self.amp_frei
        
        # in state wird die Ampliutdenabweichung gespeichert
        # positiv = aktuelle Amplitude ist größer als die Sollamplitude
        #      -> Tisch muss näher rangefahren werden
        # negativ = aktuelle Amplitude ist kleiner als die Sollamplitude
        #      -> Tisch muss weiter weggefahren werden
        self.state.append(amp - self.amp_soll) 
        self.i += 1

        # calculate reward
        # lineare Funktion - andere benutzen?
        reward = self.calc_reward()

        # scan end check
        if self.scan_lenth <= 0:
            done = True
        else:
            done = False

        info = {}

        return np.array(self.state), reward, done, info

    def load_line(self):
        self.top = np.loadtxt(self.line_file, delimiter=',')/1
        self.top_orig = self.top

    def set_line_file(self, line_file):
        self.line_file = 'lines/' + line_file 
        self.line_file = line_file 
        self.load_line()
        
    def get_line_file(self):
        return self.line_file
    
    
    def calc_reward(self):
        # hier können später verschiedene Funktionen implementiert werden
        # jetzt gerade ist es noch die lineare Funktion
        # reward = ((self.amp_soll - abs(self.state[-1])))/(10**2)
        
        # s1=self.state[-1]
        # s2=self.state[-2]
        # s3=self.state[-3]
        # s4=self.state[-4]
        # s5=self.state[-5]
        
        punishstd=1500*((np.std(self.state))**(1/2))
        punishabssum=100*((abs(self.state[-1])+abs(self.state[-2])+abs(self.state[-3])+abs(self.state[-4])+abs(self.state[-5]))/5)**2
        reward_now = ((self.amp_soll - abs(self.state[-1]))**4)#-0.2*abs(self.state[-2]-self.state[-1])**5 #/(10**7)
        reward=reward_now-punishstd-punishabssum  #*self.rewS
        reward=reward/10000;
        if reward<0:
            reward=0
        self.rewS = (reward_now/10**4)**2
        return reward
        

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    

