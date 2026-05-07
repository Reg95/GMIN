import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 10))
    radius = 9.5
    notation_size = 20
    '''0 - 10'''
    # BSRN-S, FSRCNN
    x = [13]
    y = [26.98]
    area = (30) * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#ffa333', edgecolors='white', linewidths=2.0)
    plt.annotate('FSRCNN', (13 + 10, 26.98 + 0.1 - 0.07), fontsize=notation_size)
    # plt.annotate('BSRN-S(Ours)', (156 - 70, 32.16 + 0.15), fontsize=notation_size)
    '''10 - 25'''
    # EFDN,FDIWN,CISNet,RFDN
    x = [276,454,381,550]
    y = [27.56,27.58,27.63,27.57]
    area = (75) * radius**2
    ax.scatter(x, y, s=area, alpha=1.0, marker='.', c='#3f7dff', edgecolors='white', linewidths=2.0)
    plt.annotate('EFDN', (276 - 30, 27.56 + 0.10 - 0.02), fontsize=notation_size)
    plt.annotate('FDIWN', (450 , 27.61), fontsize=notation_size)
    plt.annotate('CISNet', (370, 27.67), fontsize=notation_size)
    plt.annotate('RFDN', (530, 27.6), fontsize=notation_size)
    '''25 - 50'''
    # IDN, IMDN, PAN
    x = [553, 715, 272]
    y = [27.41, 27.56, 27.59]
    area = (140) * radius**2
    ax.scatter(x, y, s=area, alpha=0.6, marker='.', c='#a0d3f2', edgecolors='white', linewidths=2.0)
    plt.annotate('IDN', (553 - 60, 27.41 + 0.15 - 0.12), fontsize=notation_size)
    plt.annotate('IMDN', (730, 27.6), fontsize=notation_size)
    plt.annotate('PAN', (272 - 70, 27.59 - 0.25 + 0.2), fontsize=notation_size)
    '''50 - 100'''
    # SRCNN, CARN, LAPAR-A
    x = [53, 1592, 659]
    y = [26.9, 27.58, 27.61]
    area = 175 * radius**2
    ax.scatter(x, y, s=area, alpha=0.8, marker='.', c='#bfd0f0', edgecolors='white', linewidths=2.0)
    plt.annotate('SRCNN', (0, 26.83), fontsize=notation_size)
    # plt.annotate('CARN', (370, 27.67), fontsize=notation_size)
    plt.annotate('LAPAR-A', (620, 27.65), fontsize=notation_size)
    '''1M+'''
    # LapSRCN, VDSR, DRRN, MemNet
    x = [502, 666, 298, 678]
    y = [27.32, 27.29, 27.38, 27.40]
    area = (250) * radius**2
    ax.scatter(x, y, s=area, alpha=0.3, marker='.', c='#94b79d', edgecolors='white', linewidths=2.0)
    plt.annotate('LapSRCN', (553 - 170, 27.41 + 0.15 - 0.3), fontsize=notation_size)
    plt.annotate('VDSR', (600 , 27.2), fontsize=notation_size)
    plt.annotate('DRRN', (298 - 65, 27.38 - 0.35 + 0.25), fontsize=notation_size)
    plt.annotate('MemNet', (678 + 15, 27.40 + 0.18 - 0.13), fontsize=notation_size)
    '''Ours marker'''
    x = [381]
    y = [27.63]
    ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=300)
    # x = [357]
    # y = [32.30]
    # ax.scatter(x, y, alpha=1.0, marker='*', c='r', s=700)

    plt.xlim(0, 800)
    plt.ylim(26.80, 27.70)
    plt.xlabel('Parameters (K)', fontsize=35)
    plt.ylabel('PSNR (dB)', fontsize=35)
    plt.title('PSNR vs. Parameters vs. Multi-Adds', fontsize=35)

    h = [
        plt.plot([], [], color=c, marker='.', ms=i, alpha=a, ls='')[0] for i, c, a in zip(
            [40, 60, 80, 95, 110], ['#ffa333', '#3f7dff', '#a0d3f2', '#bfd0f0', '#94b79d'], [0.8, 1.0, 0.6, 0.8, 0.3])
    ]
    ax.legend(
        labelspacing=0,
        handles=h,
        handletextpad=1.0,
        markerscale=1.0,
        fontsize=17,
        title='Multi-Adds',
        title_fontsize=25,
        labels=['<10k', '10k-25k', '25k-50k', '50k-100k', '1M+'],
        scatteryoffsets=[0.0],
        loc='lower right',
        ncol=5,
        shadow=False,
        handleheight=6,
        ).get_frame().set_alpha(0.4)  # Set the transparency to 0

    for size in ax.get_xticklabels():  # Set fontsize for x-axis
        size.set_fontsize('30')
    for size in ax.get_yticklabels():  # Set fontsize for y-axis
        size.set_fontsize('30')

    ax.grid(linestyle='-.', linewidth=0.5)
    plt.show()

    fig.savefig('model_complexity_cmp_bsrn.png')


if __name__ == '__main__':
    main()