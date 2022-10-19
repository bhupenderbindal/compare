# import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np



class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims, fc6_dims, fc7_dims, fc8_dims, fc9_dims, fc10_dims, 
                 fc11_dims, fc12_dims, fc13_dims, fc14_dims, fc15_dims, fc16_dims, fc17_dims, fc18_dims, fc19_dims, fc20_dims, n_actions):
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
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
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


class DDQNAgent(object):
    def __init__(self, gamma, lr, input_dims, n_actions,
                 mem_size, batch_size, replace):
        
        self.gamma = gamma                              # ?? wird bei .learn() eingesetzt
        self.lr = lr                                    # Learning Rate
        self.input_dims = input_dims                    # Dimensionen des Inputs
        self.n_actions = n_actions                      # Anzahl der Aktionen (Ausgabeschicht)
        self.mem_size = mem_size                        # Größe des Speichers
        self.batch_size = batch_size                    # ?? Danach wird das Netz ersetzt?
        self.replace_target_cnt = replace               # ?? =100
        self.reset()                                    # Ruft die Reset Funktion auf
    
    def reset(self):
        self.action_space = [i for i in range(self.n_actions)]                  #
        self.learn_step_counter = 0                     # Zähler zum Nachvollziehen des Lernens?

        self.memoryPID = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)
        self.memoryKI = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)
        # self.q_eval = DeepQNetwork(n_actions=self.n_actions,
        #                            input_dims=self.input_dims,
        #                            fc1_dims=2100, fc2_dims=2100)
        # self.q_next = DeepQNetwork(n_actions=self.n_actions,
        #                            input_dims=self.input_dims,
        #                            fc1_dims=2100, fc2_dims=2100)
        
        # self.q_eval = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=2100, fc2_dims=2100, fc3_dims=2100)
        # self.q_next = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=2100, fc2_dims=2100, fc3_dims=2100)
        
        # self.q_eval = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100)
        # self.q_next = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100)
        
        # self.q_eval = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100, fc5_dims=100, fc6_dims=100, fc7_dims=100)
        # self.q_next = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100, fc5_dims=100, fc6_dims=100, fc7_dims=100)
        
        # self.q_eval = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100, fc5_dims=100, 
        #                             fc6_dims=100, fc7_dims=100, fc8_dims=100, fc9_dims=100, fc10_dims=100,
        #                             fc11_dims=100, fc12_dims=100, fc13_dims=100, fc14_dims=100, fc15_dims=100, 
        #                             fc16_dims=100, fc17_dims=100, fc18_dims=100, fc19_dims=100, fc20_dims=100,
        #                             fc21_dims=100, fc22_dims=100, fc23_dims=100, fc24_dims=100, fc25_dims=100, 
        #                             fc26_dims=100, fc27_dims=100, fc28_dims=100, fc29_dims=100, fc30_dims=100)
        # self.q_next = DeepQNetwork(n_actions=self.n_actions,
        #                             input_dims=self.input_dims,
        #                             fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100, fc5_dims=100, 
        #                             fc6_dims=100, fc7_dims=100, fc8_dims=100, fc9_dims=100, fc10_dims=100,
        #                             fc11_dims=100, fc12_dims=100, fc13_dims=100, fc14_dims=100, fc15_dims=100, 
        #                             fc16_dims=100, fc17_dims=100, fc18_dims=100, fc19_dims=100, fc20_dims=100,
        #                             fc21_dims=100, fc22_dims=100, fc23_dims=100, fc24_dims=100, fc25_dims=100, 
        #                             fc26_dims=100, fc27_dims=100, fc28_dims=100, fc29_dims=100, fc30_dims=100)
        
        
        self.q_eval = DeepQNetwork(n_actions=self.n_actions,
                                    input_dims=self.input_dims,
                                    fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100, fc5_dims=100, 
                                    fc6_dims=100, fc7_dims=100, fc8_dims=100, fc9_dims=100, fc10_dims=100,
                                    fc11_dims=100, fc12_dims=100, fc13_dims=100, fc14_dims=100, fc15_dims=100, 
                                    fc16_dims=100, fc17_dims=100, fc18_dims=100, fc19_dims=100, fc20_dims=100)
        self.q_next = DeepQNetwork(n_actions=self.n_actions,
                                    input_dims=self.input_dims,
                                    fc1_dims=100, fc2_dims=100, fc3_dims=100, fc4_dims=100, fc5_dims=100, 
                                    fc6_dims=100, fc7_dims=100, fc8_dims=100, fc9_dims=100, fc10_dims=100,
                                    fc11_dims=100, fc12_dims=100, fc13_dims=100, fc14_dims=100, fc15_dims=100, 
                                    fc16_dims=100, fc17_dims=100, fc18_dims=100, fc19_dims=100, fc20_dims=100)
        
        
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=self.lr, weight_decay=0.001)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,0.999)
        self.loss = nn.MSELoss()
        
        
        
    def store_transitionPID(self, state, action, reward, state_, done):
        self.memoryPID.store_transition(state, action, reward, state_, done)
        
    # def store_transitionKI(self, state, action, reward, state_, done):
    #     self.memoryKI.store_transition(state, action, reward, state_, done)

    def sample_memoryPID(self):
        state, action, reward, new_state, done = self.memoryPID.sample_buffer(self.batch_size)

        states = T.tensor(state)
        rewards = T.tensor(reward)
        dones = T.tensor(done)
        actions = T.tensor(action)
        states_ = T.tensor(new_state)

        return states, actions, rewards, states_, dones
    
    # def sample_memoryKI(self):
    #     state, action, reward, new_state, done = self.memoryKI.sample_buffer(self.batch_size)

    #     states = T.tensor(state)
    #     rewards = T.tensor(reward)
    #     dones = T.tensor(done)
    #     actions = T.tensor(action)
    #     states_ = T.tensor(new_state)

        return states, actions, rewards, states_, dones

    def pid_action(self,observation):
        res=0.15
        n_actions=self.n_actions
        
        pid_p = 5 * observation[4]
        pid_i = 0.001 * sum(observation)
        pid_d = 2.8 * (observation[4]-observation[3])
        
        pid_action = pid_p + pid_i + pid_d
        
        action = (pid_action/res) + ((n_actions + 1)/2)
        action = round(action)
        if action > (self.n_actions-1): action = self.n_actions-1
        if action < 0 : action = 0
        
        return action

    def choose_action(self, observation,pid_counter):
        
        # action_pid = self.pid_action(observation)
        state = T.tensor([observation], dtype=T.float)
        # actions = self.q_eval.forward(state)
        
        # action_ki = T.argmax(actions).item()
        
        if pid_counter>0:
            action=self.pid_action(observation)
            pid_counter=pid_counter+1
            flag_act=2
        elif abs(observation[2])>5 and (abs(observation[4])>8 and abs(observation[3])>7):
        # elif (abs(observation[4])>8 and abs(observation[3])>7):
            pid_counter=pid_counter+1
            action=self.pid_action(observation)
            flag_act=2   
        else:
            actions = self.q_eval.forward(state)
            
            # ma=20
            
            # qualitys_orig=actions.detach().numpy()
            # qualitys =np.convolve(qualitys_orig[0,:], np.ones(ma), 'same') / ma
            # qualdist=abs(qualitys_orig-qualitys)
            
            action=T.argmax(actions).item()
            flag_act=1
            if np.random.random()<0.05:
                action=action+2*np.random.randn()
            # if np.random.random()<0.5:
            #     action=T.argmax(actions).item()
            # else:
            #     action=qualitys.argmax()
            # if np.random.random()<0.01:
            #     action=qualdist.argmax()
            if np.random.random()<0.05:
                action=(self.n_actions-1)*np.random.rand()
                
                   
        
        # if np.random.random()<0.05:
        #     action=action+2*np.random.normal()
        
            
        action = round(action)
        
        if action > (self.n_actions-1): action = self.n_actions-1
        if action < 0 : action = 0
        
        act = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            act[i] = 1 if i==action else 0
            
        # self.decrement_pid_epsilon()

        return action, act, flag_act, pid_counter
    
    

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            # wird nur alle 100 Lernschritte ausgeführt
            self.q_next.load_state_dict(self.q_eval.state_dict())


    def learn(self):
        if self.memoryPID.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        # alle 100 schritte wird es einmal ersetzt
        # siehe oben mit dem Modulo rechnen
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memoryPID()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        # q_pred = self.q_eval.forward(states)[actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)
        

        
        max_actions = T.argmax(q_eval, dim=1)
        # max_actions = q_eval.argmax()
        
        # print(max_actions)
        # print(max_actions.size)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.loss(q_target, q_pred)
        # Gradienten werden durch backward berechnet
        loss.backward()

        self.optimizer.step()
        
        
        # if self.memoryKI.mem_cntr > self.batch_size:

        #     self.optimizer.zero_grad()

        #     states, actions, rewards, states_, dones = self.sample_memoryKI()
    
        #     indices = np.arange(self.batch_size)
    
        #     q_pred = self.q_eval.forward(states)[indices, actions]
        #     q_next = self.q_next.forward(states_)
        #     q_eval = self.q_eval.forward(states_)
    
            
        #     max_actions = T.argmax(q_eval, dim=1)
        #     # print(max_actions)
        #     # print(max_actions.size)
        #     q_next[dones] = 0.0
    
        #     q_target = rewards + self.gamma * q_next[indices, max_actions]
        #     loss = self.loss(q_target, q_pred)
        #     # Gradienten werden durch backward berechnet
        #     loss.backward()
    
        #     self.optimizer.step()
        
        self.learn_step_counter += 1


    def save(self, savepath):
        T.save(self.q_eval.state_dict(), savepath)
        
        
# =============================================================================
# =============================================================================
#               GET und SET Methoden
# =============================================================================
    # def get_pid_epsilon(self):
    #     pid_lr = self.pid_lr 
    #     return pid_lr
    
    # def set_pid_epsilon(self, pid_lr):
    #     self.pid_lr = pid_lr
        
    # def set_pid_epsilon_dec(self, pid_dec):
    #     self.pid_epsilon_dec = pid_dec

    # def get_n_action_pid(self):
    #     n_action_pid = self.n_action_pid
    #     return n_action_pid
    
    # def set_n_action_pid(self, n_action):
    #     self.n_action_pid = n_action
        
    # def decrement_pid_epsilon(self):
    #     self.pid_epsilon = self.pid_epsilon - self.pid_eps_dec \
    #         if self.pid_epsilon > self.pid_eps_min else 0
    #             # hier sollte noch das eps_min geändert werden

    # def decrement_epsilon(self):
    #     self.epsilon = self.epsilon - self.eps_dec \
    #         if self.epsilon > self.eps_min else self.eps_min
                        
    # def set_epsilon(self, eps):
    #     self.epsilon = eps
        
# =============================================================================





class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size  # replace old with new Memory

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # Nimmt aus dem bisherigen Speicher batch_size-viele Einträge raus
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        # übergibt die Random Aktionen aus dem Speicher an den Agenten
        return states, actions, rewards, states_, dones