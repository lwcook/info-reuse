#!/usr/bin/python
import matplotlib.pyplot as plt
import json
import subprocess
import datetime

import numpy as np

blue = [136./255., 186./255., 235./255.]
orange = [253./255., 174./255., 97./255.]
red = [140./255., 20./255., 32./255.]
green = [171./255., 221./255., 164./255.]

def main():

    date = '14May'

    dbIR, dbMC, dbIRc, dbMCc = [], [], [], []
    with open('output/Test_log_IR_objective_'+date+'.txt', 'r') as f:
        for line in f:
            dbIR.append(json.loads(line))

    with open('output/Test_log_IR_constraint_'+date+'.txt', 'r') as f:
        for line in f:
            dbIRc.append(json.loads(line))

    with open('output/Test_log_MC_objective_'+date+'.txt', 'r') as f:
        for line in f:
            dbMC.append(json.loads(line))

    with open('output/Test_log_MC_constraint_'+date+'.txt', 'r') as f:
        for line in f:
            dbMCc.append(json.loads(line))

    figsize = [5.0, 3.5]
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    ax1b = ax1.twinx()
    xplot = [sum([i['samples']/1000. for i in dbIR[0:j]]) for j in range(len(dbIR))]
    yplot = [min([i['dhat'] for i in dbIR[0:j+1]])
            for j in range(len(dbIR))]
    ax1.plot(xplot, yplot, color=blue)

    xplot = [sum([i['samples']/1000. for i in dbIRc[0:j]]) for j in
            range(len(dbIRc))]

    ax1b.plot([], [], color=blue, label='gIR objective')
    ax1b.plot(xplot, [i['dhat'] for i in dbIRc], color=blue, 
            dashes=[4, 3], label='gIR constraint')

    xplot = [sum([i['samples']/1000. for i in dbMC[0:j]]) for j in range(len(dbMC))]
    yplot = [min([i['dhat'] for i in dbMC[0:j+1]])
            for j in range(len(dbMC))]
    ax1.plot(xplot, yplot, color=red)

    xplot = [sum([i['samples']/1000. for i in dbMCc[0:j]]) for j in
            range(len(dbMCc))]
    ax1b.plot([], [], color=red, label='MC objective')
    ax1b.plot(xplot, [i['dhat'] for i in dbMCc], color=red,
            dashes=[4, 3], label='MC constraint')
    ax1.set_xlabel('Total No. of Evaluations [$10^3$]')
    ax1.set_ylabel('Objective')
    ax1b.set_ylabel('Constraint')
    #ax1b.set_ylim([-0.4, 0.6])
    ax1b.legend(loc='upper right')
    plt.tight_layout()

    savefig('Alg_totalsamples')



#    ax2 = fig1.add_subplot(2, 2, 2)
    fig, ax2 = plt.subplots(1, 1, figsize=figsize)
    xplot = [i for i in range(len(dbIR))]
    ax2.plot(xplot, [i['samples'] for i in dbIR], color=blue,
        label='gIR objective')

    xplot = [i for i in range(len(dbIRc))]
    ax2.plot(xplot, [i['samples'] for i in dbIRc], color=blue,
            dashes=[4, 3], label='gIR constraint')

    xplot = [i for i in range(len(dbMC))]
    ax2.plot(xplot, [i['samples'] for i in dbMC], color=red,
            label='MC objective')

    xplot = [i for i in range(len(dbMCc))]
    ax2.plot(xplot, [i['samples'] for i in dbMCc], color=red, 
            dashes=[4, 3], label='MC constraint')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('No. of Evaluations')
    #ax2.set_ylim([0, 700])
    plt.tight_layout()
    savefig('Alg_iterationsamples')


    xlim = [-1.1, 1.1]
    ylim = [-1.1, 1.1]

    #ax3 = fig1.add_subplot(2, 2, 3)
    fig, ax3 = plt.subplots(1, 1, figsize=figsize)
    xplot = [i['design'][0] for i in dbIR]
    yplot = [i['design'][1] for i in dbIR]
    num = len(dbIR)
    scale_plot = [max(min((float(j)/num), 1), 0) for j in range(num)]
    cb, cr = blue, red
    cplot = [[cr[0]*s+cb[0]*(1-s), cr[1]*s+cb[1]*(1-s), cr[2]*s+cb[2]*(1-s)]
        for s in scale_plot]
    ax3.scatter(xplot, yplot, c=cplot, lw=0, s=20)
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    plt.tight_layout()
    savefig('Alg_designsIR')

    fig, ax4 = plt.subplots(1, 1, figsize=figsize)

    xplot = [i['design'][0] for i in dbMC]
    yplot = [i['design'][1] for i in dbMC]
    num = len(dbMC)
    scale_plot = [max(min((float(j)/num), 1), 0) for j in range(num)]
    cb, cr = blue, red
    cplot = [[cr[0]*s+cb[0]*(1-s), cr[1]*s+cb[1]*(1-s), cr[2]*s+cb[2]*(1-s)]
        for s in scale_plot]
    ax4.scatter(xplot, yplot, c=cplot, lw=0, s=20)
    ax4.set_xlabel('$x_1$')
    ax4.set_ylabel('$x_2$')
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    plt.tight_layout()
    savefig('Alg_designsMC')


    plt.show()


def savefig(name='saved_fig', formatstr='pdf'):

    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    subprocess.call(["mkdir", "-p", "./figs/"])
    plt.savefig('./figs/' +  str(name) + '_' + str(date.day) +
                months[date.month-1] + '.' + formatstr, format=formatstr)


if __name__ == "__main__":
    main()
