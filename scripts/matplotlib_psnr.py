import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

# 打开工作表
with open("C:/Users/ASUS/Desktop/v3.CSV", "r") as file:
    list = file.readlines()
    # print(list)

# 去换行
for i, j in enumerate(list):
    list[i] = j[:-1]
# print(list)

# 拆分成列表
psnr_list1 = []
tier_list1 = []
for i in list:
    temp_list = i.split(",")
    psnr_list1.append(float(temp_list[1]))
    tier_list1.append(int(temp_list[2]))


tier_list1 = np.array(tier_list1) / 1000
print(psnr_list1, "\n", tier_list1)



# 示例
data1 = [{'Model': 'MGRNv1', 'PSNR': '31.5053', 'Iteration': '5000'}, {'Model': 'MGRNv1', 'PSNR': '31.7453', 'Iteration': '10000'}, {'Model': 'MGRNv1', 'PSNR': '31.7784', 'Iteration': '15000'}, {'Model': 'MGRNv1', 'PSNR': '31.8605', 'Iteration': '20000'}, {'Model': 'MGRNv1', 'PSNR': '31.9252', 'Iteration': '25000'}, {'Model': 'MGRNv1', 'PSNR': '31.9122', 'Iteration': '30000'}, {'Model': 'MGRNv1', 'PSNR': '31.9468', 'Iteration': '35000'}, {'Model': 'MGRNv1', 'PSNR': '31.9990', 'Iteration': '40000'}, {'Model': 'MGRNv1', 'PSNR': '31.9173', 'Iteration': '45000'}, {'Model': 'MGRNv1', 'PSNR': '31.9943', 'Iteration': '50000'}, {'Model': 'MGRNv1', 'PSNR': '32.0382', 'Iteration': '55000'}, {'Model': 'MGRNv1', 'PSNR': '32.0521', 'Iteration': '60000'}, {'Model': 'MGRNv1', 'PSNR': '31.9748', 'Iteration': '65000'}, {'Model': 'MGRNv1', 'PSNR': '32.0432', 'Iteration': '70000'}, {'Model': 'MGRNv1', 'PSNR': '32.0586', 'Iteration': '75000'}, {'Model': 'MGRNv1', 'PSNR': '32.0449', 'Iteration': '80000'}, {'Model': 'MGRNv1', 'PSNR': '32.0619', 'Iteration': '85000'}, {'Model': 'MGRNv1', 'PSNR': '32.0096', 'Iteration': '90000'}, {'Model': 'MGRNv1', 'PSNR': '32.0929', 'Iteration': '95000'}, {'Model': 'MGRNv1', 'PSNR': '32.1209', 'Iteration': '100000'}, {'Model': 'MGRNv1', 'PSNR': '32.0366', 'Iteration': '105000'}, {'Model': 'MGRNv1', 'PSNR': '32.0567', 'Iteration': '110000'}, {'Model': 'MGRNv1', 'PSNR': '32.0823', 'Iteration': '115000'}, {'Model': 'MGRNv1', 'PSNR': '32.1248', 'Iteration': '120000'}, {'Model': 'MGRNv1', 'PSNR': '32.0755', 'Iteration': '125000'}, {'Model': 'MGRNv1', 'PSNR': '32.0822', 'Iteration': '130000'}, {'Model': 'MGRNv1', 'PSNR': '32.1155', 'Iteration': '135000'}, {'Model': 'MGRNv1', 'PSNR': '32.1479', 'Iteration': '140000'}, {'Model': 'MGRNv1', 'PSNR': '32.0974', 'Iteration': '145000'}, {'Model': 'MGRNv1', 'PSNR': '32.0604', 'Iteration': '150000'}, {'Model': 'MGRNv1', 'PSNR': '32.1648', 'Iteration': '155000'}, {'Model': 'MGRNv1', 'PSNR': '32.0921', 'Iteration': '160000'}, {'Model': 'MGRNv1', 'PSNR': '32.0777', 'Iteration': '165000'}, {'Model': 'MGRNv1', 'PSNR': '32.1544', 'Iteration': '170000'}, {'Model': 'MGRNv1', 'PSNR': '32.1334', 'Iteration': '175000'}, {'Model': 'MGRNv1', 'PSNR': '32.0782', 'Iteration': '180000'}, {'Model': 'MGRNv1', 'PSNR': '32.1192', 'Iteration': '185000'}, {'Model': 'MGRNv1', 'PSNR': '32.1104', 'Iteration': '190000'}, {'Model': 'MGRNv1', 'PSNR': '32.1121', 'Iteration': '195000'}, {'Model': 'MGRNv1', 'PSNR': '32.1822', 'Iteration': '200000'}, {'Model': 'MGRNv1', 'PSNR': '32.2104', 'Iteration': '205000'}, {'Model': 'MGRNv1', 'PSNR': '32.0115', 'Iteration': '210000'}, {'Model': 'MGRNv1', 'PSNR': '32.1183', 'Iteration': '215000'}, {'Model': 'MGRNv1', 'PSNR': '32.1311', 'Iteration': '220000'}, {'Model': 'MGRNv1', 'PSNR': '32.1553', 'Iteration': '225000'}, {'Model': 'MGRNv1', 'PSNR': '31.9877', 'Iteration': '230000'}, {'Model': 'MGRNv1', 'PSNR': '32.0868', 'Iteration': '235000'}, {'Model': 'MGRNv1', 'PSNR': '32.1349', 'Iteration': '240000'}, {'Model': 'MGRNv1', 'PSNR': '32.1690', 'Iteration': '245000'}, {'Model': 'MGRNv1', 'PSNR': '32.0567', 'Iteration': '250000'}, {'Model': 'MGRNv1', 'PSNR': '32.1316', 'Iteration': '255000'}, {'Model': 'MGRNv1', 'PSNR': '32.1639', 'Iteration': '260000'}, {'Model': 'MGRNv1', 'PSNR': '32.1238', 'Iteration': '265000'}, {'Model': 'MGRNv1', 'PSNR': '32.1710', 'Iteration': '270000'}, {'Model': 'MGRNv1', 'PSNR': '32.1705', 'Iteration': '275000'}, {'Model': 'MGRNv1', 'PSNR': '32.1664', 'Iteration': '280000'}, {'Model': 'MGRNv1', 'PSNR': '32.1710', 'Iteration': '285000'}, {'Model': 'MGRNv1', 'PSNR': '32.1782', 'Iteration': '290000'}, {'Model': 'MGRNv1', 'PSNR': '32.1672', 'Iteration': '295000'}, {'Model': 'MGRNv1', 'PSNR': '32.1483', 'Iteration': '300000'}, {'Model': 'MGRNv1', 'PSNR': '32.0984', 'Iteration': '305000'}, {'Model': 'MGRNv1', 'PSNR': '32.1862', 'Iteration': '310000'}, {'Model': 'MGRNv1', 'PSNR': '32.1873', 'Iteration': '315000'}, {'Model': 'MGRNv1', 'PSNR': '32.1678', 'Iteration': '320000'}, {'Model': 'MGRNv1', 'PSNR': '32.1375', 'Iteration': '325000'}, {'Model': 'MGRNv1', 'PSNR': '32.1951', 'Iteration': '330000'}, {'Model': 'MGRNv1', 'PSNR': '32.1681', 'Iteration': '335000'}, {'Model': 'MGRNv1', 'PSNR': '32.1715', 'Iteration': '340000'}, {'Model': 'MGRNv1', 'PSNR': '32.1747', 'Iteration': '345000'}, {'Model': 'MGRNv1', 'PSNR': '32.1484', 'Iteration': '350000'}, {'Model': 'MGRNv1', 'PSNR': '32.0637', 'Iteration': '355000'}, {'Model': 'MGRNv1', 'PSNR': '32.1583', 'Iteration': '360000'}, {'Model': 'MGRNv1', 'PSNR': '32.1915', 'Iteration': '365000'}, {'Model': 'MGRNv1', 'PSNR': '32.1672', 'Iteration': '370000'}, {'Model': 'MGRNv1', 'PSNR': '32.1878', 'Iteration': '375000'}, {'Model': 'MGRNv1', 'PSNR': '32.1877', 'Iteration': '380000'}, {'Model': 'MGRNv1', 'PSNR': '32.2022', 'Iteration': '385000'}, {'Model': 'MGRNv1', 'PSNR': '32.1720', 'Iteration': '390000'}, {'Model': 'MGRNv1', 'PSNR': '32.2212', 'Iteration': '395000'}, {'Model': 'MGRNv1', 'PSNR': '32.1905', 'Iteration': '400000'}, {'Model': 'MGRNv1', 'PSNR': '32.1791', 'Iteration': '405000'}, {'Model': 'MGRNv1', 'PSNR': '32.1972', 'Iteration': '410000'}, {'Model': 'MGRNv1', 'PSNR': '32.1935', 'Iteration': '415000'}, {'Model': 'MGRNv1', 'PSNR': '32.1849', 'Iteration': '420000'}, {'Model': 'MGRNv1', 'PSNR': '32.1775', 'Iteration': '425000'}, {'Model': 'MGRNv1', 'PSNR': '32.1726', 'Iteration': '430000'}, {'Model': 'MGRNv1', 'PSNR': '32.1787', 'Iteration': '435000'}, {'Model': 'MGRNv1', 'PSNR': '32.1527', 'Iteration': '440000'}, {'Model': 'MGRNv1', 'PSNR': '32.1922', 'Iteration': '445000'}, {'Model': 'MGRNv1', 'PSNR': '32.2203', 'Iteration': '450000'}, {'Model': 'MGRNv1', 'PSNR': '32.2159', 'Iteration': '455000'}, {'Model': 'MGRNv1', 'PSNR': '32.2126', 'Iteration': '460000'}, {'Model': 'MGRNv1', 'PSNR': '32.1819', 'Iteration': '465000'}, {'Model': 'MGRNv1', 'PSNR': '32.2219', 'Iteration': '470000'}, {'Model': 'MGRNv1', 'PSNR': '32.2231', 'Iteration': '475000'}, {'Model': 'MGRNv1', 'PSNR': '32.1925', 'Iteration': '480000'}, {'Model': 'MGRNv1', 'PSNR': '32.1937', 'Iteration': '485000'}, {'Model': 'MGRNv1', 'PSNR': '32.2129', 'Iteration': '490000'}, {'Model': 'MGRNv1', 'PSNR': '32.1862', 'Iteration': '495000'}, {'Model': 'MGRNv1', 'PSNR': '32.2099', 'Iteration': '500000'}, {'Model': 'MGRNv1', 'PSNR': '32.1926', 'Iteration': '505000'}, {'Model': 'MGRNv1', 'PSNR': '32.1909', 'Iteration': '510000'}, {'Model': 'MGRNv1', 'PSNR': '32.2341', 'Iteration': '515000'}, {'Model': 'MGRNv1', 'PSNR': '32.2213', 'Iteration': '520000'},
         # {'Model': 'MGRNv1', 'PSNR': '30.9645', 'Iteration': '525000'},
         {'Model': 'MGRNv1', 'PSNR': '32.2291', 'Iteration': '530000'}, {'Model': 'MGRNv1', 'PSNR': '32.2098', 'Iteration': '535000'}, {'Model': 'MGRNv1', 'PSNR': '32.2252', 'Iteration': '540000'}, {'Model': 'MGRNv1', 'PSNR': '32.2132', 'Iteration': '545000'}, {'Model': 'MGRNv1', 'PSNR': '32.2204', 'Iteration': '550000'}, {'Model': 'MGRNv1', 'PSNR': '32.2294', 'Iteration': '555000'}, {'Model': 'MGRNv1', 'PSNR': '32.2182', 'Iteration': '560000'}, {'Model': 'MGRNv1', 'PSNR': '32.2224', 'Iteration': '565000'}, {'Model': 'MGRNv1', 'PSNR': '32.2278', 'Iteration': '570000'}, {'Model': 'MGRNv1', 'PSNR': '32.2227', 'Iteration': '575000'}, {'Model': 'MGRNv1', 'PSNR': '32.2482', 'Iteration': '580000'}, {'Model': 'MGRNv1', 'PSNR': '32.2307', 'Iteration': '585000'}, {'Model': 'MGRNv1', 'PSNR': '32.2170', 'Iteration': '590000'}, {'Model': 'MGRNv1', 'PSNR': '32.1891', 'Iteration': '595000'}, {'Model': 'MGRNv1', 'PSNR': '32.2850', 'Iteration': '600000'}, {'Model': 'MGRNv1', 'PSNR': '32.2409', 'Iteration': '605000'}, {'Model': 'MGRNv1', 'PSNR': '32.2403', 'Iteration': '610000'}, {'Model': 'MGRNv1', 'PSNR': '32.2321', 'Iteration': '615000'}, {'Model': 'MGRNv1', 'PSNR': '32.2653', 'Iteration': '620000'}, {'Model': 'MGRNv1', 'PSNR': '32.2540', 'Iteration': '625000'}, {'Model': 'MGRNv1', 'PSNR': '32.2723', 'Iteration': '630000'}, {'Model': 'MGRNv1', 'PSNR': '32.2653', 'Iteration': '635000'}, {'Model': 'MGRNv1', 'PSNR': '32.2291', 'Iteration': '640000'}, {'Model': 'MGRNv1', 'PSNR': '32.2701', 'Iteration': '645000'}, {'Model': 'MGRNv1', 'PSNR': '32.2743', 'Iteration': '650000'}, {'Model': 'MGRNv1', 'PSNR': '32.2350', 'Iteration': '655000'}, {'Model': 'MGRNv1', 'PSNR': '32.2786', 'Iteration': '660000'}, {'Model': 'MGRNv1', 'PSNR': '32.2515', 'Iteration': '665000'}, {'Model': 'MGRNv1', 'PSNR': '32.2730', 'Iteration': '670000'}, {'Model': 'MGRNv1', 'PSNR': '32.2470', 'Iteration': '675000'}, {'Model': 'MGRNv1', 'PSNR': '32.2575', 'Iteration': '680000'}, {'Model': 'MGRNv1', 'PSNR': '32.2652', 'Iteration': '685000'}, {'Model': 'MGRNv1', 'PSNR': '32.2545', 'Iteration': '690000'}, {'Model': 'MGRNv1', 'PSNR': '32.2710', 'Iteration': '695000'}, {'Model': 'MGRNv1', 'PSNR': '32.2512', 'Iteration': '700000'}, {'Model': 'MGRNv1', 'PSNR': '32.2712', 'Iteration': '705000'}, {'Model': 'MGRNv1', 'PSNR': '32.2717', 'Iteration': '710000'}, {'Model': 'MGRNv1', 'PSNR': '32.2525', 'Iteration': '715000'}, {'Model': 'MGRNv1', 'PSNR': '32.2778', 'Iteration': '720000'}, {'Model': 'MGRNv1', 'PSNR': '32.2886', 'Iteration': '725000'}, {'Model': 'MGRNv1', 'PSNR': '32.2604', 'Iteration': '730000'}, {'Model': 'MGRNv1', 'PSNR': '32.2923', 'Iteration': '735000'}, {'Model': 'MGRNv1', 'PSNR': '32.2620', 'Iteration': '740000'}, {'Model': 'MGRNv1', 'PSNR': '32.2714', 'Iteration': '745000'}, {'Model': 'MGRNv1', 'PSNR': '32.2888', 'Iteration': '750000'}, {'Model': 'MGRNv1', 'PSNR': '32.2868', 'Iteration': '755000'}, {'Model': 'MGRNv1', 'PSNR': '32.2941', 'Iteration': '760000'}, {'Model': 'MGRNv1', 'PSNR': '32.2861', 'Iteration': '765000'}, {'Model': 'MGRNv1', 'PSNR': '32.2932', 'Iteration': '770000'}, {'Model': 'MGRNv1', 'PSNR': '32.2811', 'Iteration': '775000'}, {'Model': 'MGRNv1', 'PSNR': '32.2959', 'Iteration': '780000'}, {'Model': 'MGRNv1', 'PSNR': '32.2741', 'Iteration': '785000'}, {'Model': 'MGRNv1', 'PSNR': '32.2887', 'Iteration': '790000'}, {'Model': 'MGRNv1', 'PSNR': '32.2909', 'Iteration': '795000'}, {'Model': 'MGRNv1', 'PSNR': '32.3015', 'Iteration': '800000'}, {'Model': 'MGRNv1', 'PSNR': '32.2887', 'Iteration': '805000'}, {'Model': 'MGRNv1', 'PSNR': '32.2770', 'Iteration': '810000'}, {'Model': 'MGRNv1', 'PSNR': '32.2767', 'Iteration': '815000'}, {'Model': 'MGRNv1', 'PSNR': '32.2887', 'Iteration': '820000'}, {'Model': 'MGRNv1', 'PSNR': '32.2928', 'Iteration': '825000'}, {'Model': 'MGRNv1', 'PSNR': '32.2857', 'Iteration': '830000'}, {'Model': 'MGRNv1', 'PSNR': '32.2929', 'Iteration': '835000'}, {'Model': 'MGRNv1', 'PSNR': '32.2972', 'Iteration': '840000'}, {'Model': 'MGRNv1', 'PSNR': '32.2810', 'Iteration': '845000'}, {'Model': 'MGRNv1', 'PSNR': '32.3003', 'Iteration': '850000'}, {'Model': 'MGRNv1', 'PSNR': '32.2877', 'Iteration': '855000'}, {'Model': 'MGRNv1', 'PSNR': '32.2878', 'Iteration': '860000'}, {'Model': 'MGRNv1', 'PSNR': '32.2941', 'Iteration': '865000'}, {'Model': 'MGRNv1', 'PSNR': '32.2900', 'Iteration': '870000'}, {'Model': 'MGRNv1', 'PSNR': '32.2977', 'Iteration': '875000'}, {'Model': 'MGRNv1', 'PSNR': '32.3020', 'Iteration': '880000'}, {'Model': 'MGRNv1', 'PSNR': '32.2918', 'Iteration': '885000'}, {'Model': 'MGRNv1', 'PSNR': '32.2966', 'Iteration': '890000'}, {'Model': 'MGRNv1', 'PSNR': '32.3044', 'Iteration': '895000'}, {'Model': 'MGRNv1', 'PSNR': '32.2942', 'Iteration': '900000'}, {'Model': 'MGRNv1', 'PSNR': '32.2920', 'Iteration': '905000'}, {'Model': 'MGRNv1', 'PSNR': '32.3059', 'Iteration': '910000'}, {'Model': 'MGRNv1', 'PSNR': '32.2956', 'Iteration': '915000'}, {'Model': 'MGRNv1', 'PSNR': '32.2986', 'Iteration': '920000'}, {'Model': 'MGRNv1', 'PSNR': '32.2909', 'Iteration': '925000'}, {'Model': 'MGRNv1', 'PSNR': '32.3002', 'Iteration': '930000'}, {'Model': 'MGRNv1', 'PSNR': '32.2963', 'Iteration': '935000'}, {'Model': 'MGRNv1', 'PSNR': '32.3015', 'Iteration': '940000'}, {'Model': 'MGRNv1', 'PSNR': '32.2996', 'Iteration': '945000'}, {'Model': 'MGRNv1', 'PSNR': '32.3017', 'Iteration': '950000'}, {'Model': 'MGRNv1', 'PSNR': '32.2982', 'Iteration': '955000'}, {'Model': 'MGRNv1', 'PSNR': '32.3026', 'Iteration': '960000'}, {'Model': 'MGRNv1', 'PSNR': '32.3038', 'Iteration': '965000'}, {'Model': 'MGRNv1', 'PSNR': '32.3020', 'Iteration': '970000'}, {'Model': 'MGRNv1', 'PSNR': '32.3012', 'Iteration': '975000'}, {'Model': 'MGRNv1', 'PSNR': '32.2984', 'Iteration': '980000'}, {'Model': 'MGRNv1', 'PSNR': '32.3028', 'Iteration': '985000'}, {'Model': 'MGRNv1', 'PSNR': '32.3019', 'Iteration': '990000'}, {'Model': 'MGRNv1', 'PSNR': '32.3017', 'Iteration': '995000'}, {'Model': 'MGRNv1', 'PSNR': '32.3021', 'Iteration': '1000000'}, {'Model': 'MGRNv1', 'PSNR': '32.3021', 'Iteration': '1005000'}]

# Convert
data_list = []
for i in data1:
    data_list.append((i["PSNR"], i["Iteration"]))
tier1 = []
psnr1 = []
for i in data_list:
    tier1.append(int(i[1]))
    psnr1.append(float(i[0]))

# print(tier)
tier1 = np.array(tier1) / 1000
# print(tier)


# 画图片
#曲线平滑处理
m = make_interp_spline(tier1, psnr1)
xs = np.linspace(0, 1000, 20000)
ys = m(xs)

m1 = make_interp_spline(tier_list1, psnr_list1)
xs1 = np.linspace(0, 1000, 50)
ys1 = m1(xs1)

y_smoothed = gaussian_filter1d(psnr1, sigma=10)

fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(tier1, psnr1, label="v1")
ax.plot(tier_list1, psnr_list1, label="v2")
ax.plot(xs, ys, label=" v1 test")
ax.plot(xs1, ys1, label=" v2 Smooth 2")  # 方式1平滑
ax.plot(tier1, y_smoothed, label="v2 Smooth 1")  # 方式2平滑

ax.legend()
ax.grid()

# 对比范围和名称的区别
# plt.xticks(np.arange(0, 1000, 100))  # 调整坐标轴刻度
# plt.yticks(np.arange(31, 32.5, 0.1))
plt.xlabel('Iterations (Kilo)')
plt.ylabel('PSNR (dB)')

# plt.xlim((0, 1000))  # 调整坐标轴
# plt.ylim((1, 200))