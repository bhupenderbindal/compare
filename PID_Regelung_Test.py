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
    target_deflection=0
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
        delta_target=round((pid_p + pid_i + pid_d)/0.15)*0.15
        target_deflection=target_deflection + delta_target
        #target_deflection=target_deflection + pid_p + pid_i + pid_d
        def_err=target_deflection-piezo[0]
        u_new = u_new + def_err*0.003
        
        
        u_list.append(u_new)
    abweichung = sum_amp_diff_quadrat / t_end
    return deflection, amp, abweichung