import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from Simulationsfunktion import simulationfunction
from collections import deque
from PID_Regelung_Test import pid_regelung

amp_soll = 10
amp_frei = 20
input_dims = 5
fc1_dims = 200
fc2_dims = 200
fc3_dims = 200
fc4_dims = 200
fc5_dims = 200
fc6_dims = 200
fc7_dims = 200
fc8_dims = 200
fc9_dims = 200
fc10_dims = 200
fc11_dims = 200
fc12_dims = 200
fc13_dims = 200
fc14_dims = 200
fc15_dims = 200
fc16_dims = 200
fc17_dims = 200
fc18_dims = 200
fc19_dims = 200
fc20_dims = 200
n_actions = 101
u_range = 0.3
# pid = [0.018, 0.006, 0.018]
# pid = [0.024, 0.0155, 0.015]
pid = [0.025, 0.012, 0.016]
savepath = 'ddqn_PID-KI_Entwicklung_60.pkl'
plotqual=1;
quali=600;
ma=10;
# savepath = 'ddqn_5_128_256_251action_0.5V_1000P_r4_64BS_0.001lr_500replace_10000EP.pkl'


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims, fc6_dims, fc7_dims, fc8_dims, fc9_dims, fc10_dims, fc11_dims, fc12_dims, fc13_dims, fc14_dims, fc15_dims, fc16_dims, fc17_dims, fc18_dims, fc19_dims, fc20_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims                        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.fc7_dims = fc7_dims
        self.fc8_dims = fc8_dims
        self.fc9_dims = fc9_dims
        self.fc10_dims = fc10_dims
        self.fc11_dims = fc11_dims
        self.fc12_dims = fc12_dims
        self.fc13_dims = fc13_dims
        self.fc14_dims = fc14_dims
        self.fc15_dims = fc15_dims
        self.fc16_dims = fc16_dims
        self.fc17_dims = fc17_dims
        self.fc18_dims = fc18_dims
        self.fc19_dims = fc19_dims
        self.fc20_dims = fc20_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        # self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # self.fc3.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        # self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        # self.fc4.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc4.weight.data.normal_(0, 0.1)  # initialization
        # self.fc5 = nn.Linear(self.fc4_dims, self.n_actions)
        # self.fc5.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.fc5.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc6 = nn.Linear(self.fc5_dims, self.fc6_dims)
        self.fc6.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc7 = nn.Linear(self.fc6_dims, self.fc7_dims)
        self.fc7.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc8 = nn.Linear(self.fc7_dims, self.fc8_dims)
        self.fc8.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc9 = nn.Linear(self.fc8_dims, self.fc9_dims)
        self.fc9.weight.data.normal_(0, 0.1)  # initialization
        
        self.fc10 = nn.Linear(self.fc9_dims, self.fc10_dims)
        self.fc10.weight.data.normal_(0, 0.1)  # initialization
        
        
        self.fc11 = nn.Linear(self.fc10_dims, self.fc11_dims)
        self.fc11.weight.data.normal_(0, 0.1)  # initialization
        self.fc12 = nn.Linear(self.fc11_dims, self.fc12_dims)
        self.fc12.weight.data.normal_(0, 0.1)  # initialization
        self.fc13 = nn.Linear(self.fc12_dims, self.fc13_dims)
        self.fc13.weight.data.normal_(0, 0.1)  # initialization
        self.fc14 = nn.Linear(self.fc13_dims, self.fc14_dims)
        self.fc14.weight.data.normal_(0, 0.1)  # initialization
        self.fc15 = nn.Linear(self.fc14_dims, self.fc15_dims)
        self.fc15.weight.data.normal_(0, 0.1)  # initialization        
        self.fc16 = nn.Linear(self.fc15_dims, self.fc16_dims)
        self.fc16.weight.data.normal_(0, 0.1)  # initialization        
        self.fc17 = nn.Linear(self.fc16_dims, self.fc17_dims)
        self.fc17.weight.data.normal_(0, 0.1)  # initialization        
        self.fc18 = nn.Linear(self.fc17_dims, self.fc18_dims)
        self.fc18.weight.data.normal_(0, 0.1)  # initialization        
        self.fc19 = nn.Linear(self.fc18_dims, self.fc19_dims)
        self.fc19.weight.data.normal_(0, 0.1)  # initialization        
        self.fc20 = nn.Linear(self.fc19_dims, self.fc20_dims)
        self.fc20.weight.data.normal_(0, 0.1)  # initialization
        
        
        self.fc21 = nn.Linear(self.fc20_dims, self.n_actions)
        self.fc21.weight.data.normal_(0, 0.1)  # initialization
        
        # device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.to(device)

        T.backends.cuda.matmul.allow_tf32 = False
        T.backends.cudnn.allow_tf32 = False

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))
        x = F.leaky_relu(self.fc7(x))
        x = F.leaky_relu(self.fc8(x))
        x = F.leaky_relu(self.fc9(x))
        x = F.leaky_relu(self.fc10(x))
        x = F.leaky_relu(self.fc11(x))
        x = F.leaky_relu(self.fc12(x))
        x = F.leaky_relu(self.fc13(x))
        x = F.leaky_relu(self.fc14(x))
        x = F.leaky_relu(self.fc15(x))
        x = F.leaky_relu(self.fc16(x))
        x = F.leaky_relu(self.fc17(x))
        x = F.leaky_relu(self.fc18(x))
        x = F.leaky_relu(self.fc19(x))
        x = F.leaky_relu(self.fc20(x))
        
        # x = self.fc1(state)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        # x = self.fc5(x)
        # x = self.fc6(x)
        # x = self.fc7(x)
        # x = self.fc8(x)
        # x = self.fc9(x)
        # x = self.fc10(x)
        
        # x = F.softplus(self.fc1(state))
        # x = F.softplus(self.fc2(x))
        # x = F.softplus(self.fc3(x))
        
        # x=self.fc1(state)
        # x = F.celu(x)+0.01*x
        # x=self.fc2(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc3(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc4(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc5(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc6(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc7(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc8(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc9(x)
        # x = F.celu(x)+0.01*x
        # x=self.fc10(x)
        # x = F.celu(x)+0.01*x
        
        actions = self.fc21(x)
        
        # qualitys=actions.detach().numpy()
        # actions =np.convolve(qualitys[0,:], np.ones(20), 'same') / 20
        
        return actions


def plot_scann_mit_regelung(amplitude, top, ki_def, delta_u, ki_mse, pid_def, pid_amp, pid_mse):
    fig, ax = plt.subplots()
    # plt.text(-5, 150, 'Amp_MSE zu soll_Amp:\n{}'.format(ki_mse))
    plt.text(-50, 0, 'KI-Regelung AMP_MSE:{}\nPID-Regelung AMP_MSE:{}'.format(ki_mse, pid_mse),
              bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)
    t_end = len(top) / 10 #+ 0.1
    time = np.arange(0, t_end, 0.1)
    ki_deflections = [ki_def[i] * -1 for i in range(len(ki_def))]
    pid_deflections = [pid_def[i] * -1 for i in range(len(pid_def))]
    ax.plot(time, ki_deflections, color='pink', label='nagetive Ausdehnung durch KI-Regelung', linewidth=1)
    ax.plot(time, pid_deflections, color='dodgerblue', label='nagetive Ausdehnung durch PID-Regelung', linewidth=1)
    ax.plot(time, top, color='g', label='Topographie', linewidth=1)
    ax.plot(time, amplitude, color='red', label='Amplitude durch KI-Regelung', linewidth=1)
    ax.plot(time, pid_amp, color='black', label='Amplitude durch PID-Regelung', linewidth=1)
    # ax1 = ax.twinx()
    # ax1.plot(time, delta_u, 'y', label='delta U', linewidth=1)
    # # ax1.set_ylim(-1, 1)
    plt.ylabel('Länge [nm]')
    plt.xlabel('Zeit[ms]')
    # plt.title('Testenscann mit Intelligenz und PID-Regelung\n(kp={}, ki={}, kd={})'.format(pid[0], pid[1], pid[2]))

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper right', fontsize='large')
    plt.show()


agent = DeepQNetwork(input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims, fc6_dims, fc7_dims, fc8_dims, fc9_dims, fc10_dims, 
                     fc11_dims, fc12_dims, fc13_dims, fc14_dims, fc15_dims, fc16_dims, fc17_dims, fc18_dims, fc19_dims, fc20_dims, n_actions)
agent.load_state_dict(T.load(savepath))
testdaten = np.loadtxt('line9.dat', delimiter=',') #*0.3
# testdaten = np.loadtxt('line1.dat', delimiter=',')/1
# testdaten=testdaten[1:500]
n_testdaten = len(testdaten)
piezo = [0, 0, 0, 0, 0, 0]
ki_def = []
delta_u_list = []
u_new = 0
amp = [amp_soll] * n_testdaten
amp_diff = deque([0, 0, 0, 0, 0], maxlen=5) 
ki_amp_mes = 0
qi=1
for i in range(n_testdaten):
    amp[i] = amp_soll - (piezo[0] + testdaten[i])
    if amp[i] < 0:
        amp[i] = 0
    if amp[i] > amp_frei:
        amp[i] = amp_frei
    amp_diff.append(amp[i]-amp_soll)
    state = T.tensor(amp_diff).float()
    actions = agent.forward(state.unsqueeze(dim=0))
    qualitys=actions.detach().numpy()
    aq =np.convolve(qualitys[0,:], np.ones(ma), 'same') / ma
    
    # https://ai.stackexchange.com/questions/18645/why-are-reinforcement-learning-methods-sample-inefficient
    steep_zone = [1111,1139,1140,1141,1142,1143,1144,1145,1171]
    #if plotqual and qi==quali:
    if plotqual and qi in steep_zone:    
        x = np.array(range(0, n_actions))
        du = (x * 2 * u_range) / (n_actions - 1) - u_range
        
        figq,axq = plt.subplots()
        axq.plot(du,qualitys[0,:], color="C0")
        axq.set_xlabel("Voltage change / V")
        axq.set_ylabel("Quality", color="C0")
        axq.tick_params(axis='x', color="C0")
        axq.tick_params(axis='y', color="C0")
        axq.plot(du,aq, color="C1")
        plt.grid()
    
        plt.show()
        #qi=0
        
    qi=qi+1
    action = T.argmax(actions).item()
    action=aq.argmax()
    delta_u_new = (action * 2 * u_range) / (n_actions - 1) - u_range
    u_new += delta_u_new
    piezo = simulationfunction(piezo[0], piezo[1], piezo[2], piezo[3], piezo[4], piezo[5], u_new)
    ki_def.append(piezo[0])
    delta_u_list.append(delta_u_new)
    ki_amp_mes += ((amp[i] - amp_soll)**2)

ki_amp_mes /= n_testdaten
pid_def, pid_amp, pid_amp_mse = pid_regelung(testdaten, amp_soll, amp_frei, pid[0], pid[1], pid[2])
plot_scann_mit_regelung(amp, testdaten, ki_def, delta_u_list, ki_amp_mes, pid_def, pid_amp, pid_amp_mse)
