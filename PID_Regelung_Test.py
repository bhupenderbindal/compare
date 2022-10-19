from Simulationsfunktion import simulationfunction
import numpy as np
import matplotlib.pyplot as plt


def pid_regelung(top_data, amp_soll, amp_frei, kp, ki, kd):
    t_end = len(top_data)
    piezo = (0, 0, 0, 0, 0, 0)
    amp = [amp_soll] * t_end
    deflection = []
    amp_diff_list = []
    u_list = []
    u_new = 0
    sum_amp_diff_quadrat = 0
    amp_diff_sum = 0
    for t in range(t_end):
        piezo = simulationfunction(piezo[0], piezo[1], piezo[2], piezo[3], piezo[4], piezo[5], u_new)
        deflection.append(piezo[0])
        amp[t] = amp_soll - (top_data[t] + deflection[t])
        if amp[t] < 0:
            amp[t] = 0
        if amp[t] > amp_frei:
            amp[t] = amp_frei
        amp_diff = amp[t] - amp_soll
        amp_diff_list.append(amp_diff)
        if t>5:
            amp_diff_sum =(amp_diff_list[t] + amp_diff_list[t - 1] + amp_diff_list[t - 2] + amp_diff_list[t - 3] + amp_diff_list[t - 4])
        else:
            amp_diff_sum += amp_diff
        # amp_diff_sum += amp_diff
        sum_amp_diff_quadrat += amp_diff ** 2
        pid_p = kp * amp_diff
        pid_i = ki * amp_diff_sum
        pid_d = kd * (amp_diff_list[t] - amp_diff_list[t - 1])
        u_new = u_new + pid_p + pid_i + pid_d
        
        # Spannung wird diskretisiert -> Anpassen auf KI Regelung 0.5/201
        u_new = round(u_new / 0.006) * 0.006
        u_list.append(u_new)
    abweichung = sum_amp_diff_quadrat / t_end
    return deflection, amp, abweichung


# x = 30
# n_flach = 100
# n_steig = 900
# asoll = 10
# afrei = 20
# pid = [0.02, 0.00005, 0.015]
# top_flach = [0] * n_flach
# top_steig = [x] * n_steig
# top = np.hstack((top_flach, top_steig))
# pid_def, pid_amp, pid_amp_mse = pid_regelung(top, asoll, afrei, pid[0], pid[1], pid[2])
#
# fig, ax = plt.subplots()
# t_end = len(top) // 10
# time = np.arange(0, t_end, 0.1)
# pid_deflections = [pid_def[i] * -1 for i in range(len(pid_def))]
# ax.plot(time, pid_deflections, color='r', label='Piezoausdehnung durch PID-Regelung', linewidth=1)
# ax.plot(time, top, color='green', label='Topographie', linewidth=1)
# plt.ylabel('Länge [nm]')
# plt.xlabel('Zeit[ms]')
#
# plt.title('Testenscann mit Intelligenz und PID-Regelung\n(kp={}, ki={}, kd={})'.format(pid[0], pid[1], pid[2]))
# plt.show()
