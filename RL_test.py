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
fc1_dims = 100
fc2_dims = 100
fc3_dims = 100
fc4_dims = 100
fc5_dims = 100
fc6_dims = 100
fc7_dims = 100
fc8_dims = 100
fc9_dims = 100
fc10_dims = 100
fc11_dims = 100
fc12_dims = 100
fc13_dims = 100
fc14_dims = 100
fc15_dims = 100
fc16_dims = 100
fc17_dims = 100
fc18_dims = 100
fc19_dims = 100
fc20_dims = 100
# fc21_dims = 100
# fc22_dims = 100
# fc23_dims = 100
# fc24_dims = 100
# fc25_dims = 100
# fc26_dims = 100
# fc27_dims = 100
# fc28_dims = 100
# fc29_dims = 100
# fc30_dims = 100
n_actions = 101
res = 0.15
# pid = [0.018, 0.006, 0.018]
# pid = [0.024, 0.0155, 0.015]
pid = [0.1, 0.000, 0.0]
# pid = [0.02, 0.013, 0.013]
savepath = 'ddqn_PID-KI_Entwicklung_20.pkl'
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
        # self.fc21_dims = fc21_dims
        # self.fc22_dims = fc22_dims
        # self.fc23_dims = fc23_dims
        # self.fc24_dims = fc24_dims
        # self.fc25_dims = fc25_dims
        # self.fc26_dims = fc26_dims
        # self.fc27_dims = fc27_dims
        # self.fc28_dims = fc28_dims
        # self.fc29_dims = fc29_dims
        # self.fc30_dims = fc30_dims
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
        
        # self.fc21 = nn.Linear(self.fc20_dims, self.fc21_dims)
        # self.fc21.weight.data.normal_(0, 0.1)  # initialization
        # self.fc22 = nn.Linear(self.fc21_dims, self.fc22_dims)
        # self.fc22.weight.data.normal_(0, 0.1)  # initialization
        # self.fc23 = nn.Linear(self.fc22_dims, self.fc23_dims)
        # self.fc23.weight.data.normal_(0, 0.1)  # initialization
        # self.fc24 = nn.Linear(self.fc23_dims, self.fc24_dims)
        # self.fc24.weight.data.normal_(0, 0.1)  # initialization
        # self.fc25 = nn.Linear(self.fc24_dims, self.fc25_dims)
        # self.fc25.weight.data.normal_(0, 0.1)  # initialization        
        # self.fc26 = nn.Linear(self.fc25_dims, self.fc26_dims)
        # self.fc26.weight.data.normal_(0, 0.1)  # initialization        
        # self.fc27 = nn.Linear(self.fc26_dims, self.fc27_dims)
        # self.fc27.weight.data.normal_(0, 0.1)  # initialization        
        # self.fc28 = nn.Linear(self.fc27_dims, self.fc28_dims)
        # self.fc28.weight.data.normal_(0, 0.1)  # initialization        
        # self.fc29 = nn.Linear(self.fc28_dims, self.fc29_dims)
        # self.fc29.weight.data.normal_(0, 0.1)  # initialization        
        # self.fc30 = nn.Linear(self.fc29_dims, self.fc30_dims)
        # self.fc30.weight.data.normal_(0, 0.1)  # initialization
        
        
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
        # x = F.leaky_relu(self.fc21(x))
        # x = F.leaky_relu(self.fc22(x))
        # x = F.leaky_relu(self.fc23(x))
        # x = F.leaky_relu(self.fc24(x))
        # x = F.leaky_relu(self.fc25(x))
        # x = F.leaky_relu(self.fc26(x))
        # x = F.leaky_relu(self.fc27(x))
        # x = F.leaky_relu(self.fc28(x))
        # x = F.leaky_relu(self.fc29(x))
        # x = F.leaky_relu(self.fc30(x))
        
        actions = self.fc21(x)
        
        # qualitys=actions.detach().numpy()
        # actions =np.convolve(qualitys[0,:], np.ones(20), 'same') / 20
        
        return actions


def plot_scann_mit_regelung(amplitude, top, ki_def, delta_u, ki_mse, pid_def, pid_amp, pid_mse):
    fig, ax = plt.subplots()
    # plt.text(-5, 100, 'Amp_MSE zu soll_Amp:\n{}'.format(ki_mse))
    plt.text(-50, 0, 'KI-Regelung AMP_MSE:{}\nPID-Regelung AMP_MSE:{}'.format(ki_mse, pid_mse),
              bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10)
    plt.grid()
    t_end = len(top) / 10 #+ 0.1
    time = np.arange(0, t_end, 0.1)
    ki_deflections = [ki_def[i] * -1 for i in range(len(ki_def))]
    pid_deflections = [pid_def[i] * -1 for i in range(len(pid_def))]
    ax.plot(time, ki_deflections, color='blue', label='negative Ausdehnung durch KI-Regelung', linewidth=1)
    ax.plot(time, pid_deflections, color='dodgerblue', label='negative Ausdehnung durch PID-Regelung', linewidth=1)
    ax.plot(time, top, color='g', label='Topographie', linewidth=1)
    ax.plot(time, amplitude, color='red', label='Amplitude durch KI-Regelung', linewidth=1)
    ax.plot(time, pid_amp, color='lightcoral', label='Amplitude durch PID-Regelung', linewidth=1)
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
testdaten = np.loadtxt('line1.dat', delimiter=',')*1
# testdaten=testdaten[1:500]
n_testdaten = len(testdaten)
piezo = [0, 0, 0, 0, 0, 0]
ki_def = []
delta_u_list = []
target_deflection=0;
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
    
    if plotqual and qi==quali:
        
        x = np.array(range(0, n_actions))
        d_t = (x -((n_actions + 1)/2))*res
        
        figq,axq = plt.subplots()
        axq.plot(d_t,qualitys[0,:], color="C0")
        axq.set_xlabel("height change / nm")
        axq.set_ylabel("Quality", color="C0")
        axq.tick_params(axis='x', color="C0")
        axq.tick_params(axis='y', color="C0")
        axq.plot(d_t,aq, color="C1")
        plt.grid()
    
        plt.show()
        qi=0
        
    qi=qi+1
    action = T.argmax(actions).item()
    action=aq.argmax()
    delta_target = (action -((n_actions + 1)/2))*res
    target_deflection += delta_target
    u_new += (target_deflection-piezo[0])*0.003
    piezo = simulationfunction(piezo[0], piezo[1], piezo[2], piezo[3], piezo[4], piezo[5], u_new)
    ki_def.append(piezo[0])
    ki_amp_mes += ((amp[i] - amp_soll)**2)

ki_amp_mes /= n_testdaten
pid_def, pid_amp, pid_amp_mse = pid_regelung(testdaten, amp_soll, amp_frei, pid[0], pid[1], pid[2])
plot_scann_mit_regelung(amp, testdaten, ki_def, delta_u_list, ki_amp_mes, pid_def, pid_amp, pid_amp_mse)
