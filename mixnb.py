import numpy as np


def mean(data):
    return np.mean(data)


def gap(data, mean):
    return data - mean


def std_dev(gap):
    return np.sqrt(np.sum(np.square(gap)) / 4)


def proba(std, mean, inp):
    return (1 / np.sqrt(2 * 3.14 * std)) * (np.exp(-((np.square(inp - mean)) / (2 * np.square(std)))))


ps_yes = [4500, 7000, 2500, 8000, 6500]
s1_yes = [3.6, 3.2, 3, 3.7, 2.8]
s2_yes = [3.3, 3.6, 2.3, 3.3, 3.2]
s3_yes = [3.1, 3.2, 3.7, 3.4, 3.15]
s4_yes = [3.8, 2.7, 3.7, 3.3, 2.8]
ipk_yes = [3.4, 3.2, 3.3, 3.45, 2.95]
ps_no = [3500, 7500, 3000, 3500, 5000]
s1_no = [3.1, 3.25, 3.4, 3.9, 3.4]
s2_no = [3.2, 2.8, 2.8, 3.3, 4]
s3_no = [3.2, 3.6, 2.1, 3.85, 2.8]
s4_no = [3.4, 3.25, 2.8, 3.8, 3.2]
ipk_no = [3.2, 3.4, 2.6, 3.8, 3.35]

# MEAN

ps_mean_yes = mean(ps_yes)
ps_mean_no = mean(ps_no)

s1_mean_yes = mean(s1_yes)
s1_mean_no = mean(s1_no)

s2_mean_yes = mean(s2_yes)
s2_mean_no = mean(s2_no)

s3_mean_yes = mean(s3_yes)
s3_mean_no = mean(s3_no)

s4_mean_yes = mean(s4_yes)
s4_mean_no = mean(s4_no)

ipk_mean_yes = mean(ipk_yes)
ipk_mean_no = mean(ipk_no)

# GAP

ps_gap_yes = gap(ps_yes, ps_mean_yes)
ps_gap_no = gap(ps_no, ps_mean_no)

s1_gap_yes = gap(s1_yes, s1_mean_yes)
s1_gap_no = gap(s1_no, s1_mean_no)

s2_gap_yes = gap(s2_yes, s2_mean_yes)
s2_gap_no = gap(s2_no, s2_mean_no)

s3_gap_yes = gap(s3_yes, s3_mean_yes)
s3_gap_no = gap(s3_no, s3_mean_no)

s4_gap_yes = gap(s4_yes, s4_mean_yes)
s4_gap_no = gap(s4_no, s4_mean_no)

ipk_gap_yes = gap(ipk_yes, ipk_mean_yes)
ipk_gap_no = gap(ipk_no, ipk_mean_no)

# STD DEV
ps_std_yes = std_dev(ps_gap_yes)
ps_std_no = std_dev(ps_gap_no)

s1_std_yes = std_dev(s1_gap_yes)
s1_std_no = std_dev(s1_gap_no)

s2_std_yes = std_dev(s2_gap_yes)
s2_std_no = std_dev(s2_gap_no)

s3_std_yes = std_dev(s3_gap_yes)
s3_std_no = std_dev(s3_gap_no)

s4_std_yes = std_dev(s4_gap_yes)
s4_std_no = std_dev(s4_gap_no)

ipk_std_yes = std_dev(ipk_gap_yes)
ipk_std_no = std_dev(ipk_gap_no)

# PROBA
ps_prob_yes = proba(ps_std_yes, ps_mean_yes, 3500)
ps_prob_no = proba(ps_std_no, ps_mean_no, 3500)

s1_prob_yes = proba(s1_std_yes, s1_mean_yes, 3.3)
s1_prob_no = proba(s1_std_no, s1_mean_no, 3.3)

s2_prob_yes = proba(s2_std_yes, s2_mean_yes, 3.4)
s2_prob_no = proba(s2_std_no, s2_mean_no, 3.4)

s3_prob_yes = proba(s3_std_yes, s3_mean_yes, 3.1)
s3_prob_no = proba(s3_std_no, s3_mean_no, 3.1)

s4_prob_yes = proba(s4_std_yes, s4_mean_yes, 3.2)
s4_prob_no = proba(s4_std_no, s4_mean_no, 3.2)

ipk_prob_yes = proba(ipk_std_yes, ipk_mean_yes, 3.3)
ipk_proba_no = proba(ipk_std_no, ipk_mean_no, 3.3)

result_multi_yes = (0.6 * 0.6 * 0.6 * 0.5)
result_multi_no = (0.4 * 0.4 * 0.6 * 0.5)

result_gauss_yes = (ps_prob_yes * s1_prob_yes * s2_prob_yes * s3_prob_yes * s4_prob_yes * ipk_prob_yes * 0.5)
result_gauss_no = (ps_prob_no * s1_prob_no * s2_prob_no * s3_prob_no * s4_prob_no * ipk_proba_no * 0.5)

result_akhir_yes = result_multi_yes * result_gauss_yes
result_akhir_no = result_multi_no * result_gauss_no

print("probabilitas multinomial Yes: {}".format(result_multi_yes))
print("probabilitas multinomial No: {}".format(result_multi_no))

print("probabilitas gaussian Yes: {}".format(result_gauss_yes))
print("probabilitas gaussian No: {}".format(result_gauss_no))

print("Hasil kali probabilitas Yes: {}".format(result_akhir_yes))
print("Hasil kali probabilitas No: {}".format(result_akhir_no))

print("Normalisasi hasil yes: {}".format(result_akhir_yes / (result_akhir_yes + result_akhir_no)))
print("Normalisasi hasil No: {}".format(result_akhir_no / (result_akhir_yes + result_akhir_no)))
