import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import seaborn as sns


sns.set_style('whitegrid')
colors = sns.color_palette()

def read_matrix(filename):
	f = open (filename , 'r')
	l = [ map(float,line.split()) for line in f ]
	return np.array(l)


def plot_bar_mmd():
	s_data = read_matrix('mmd2.txt')[:,2:]
	t_data = read_matrix('mmd.txt')
	t_labels = t_data[:,1]
	t_preds = (t_data[:,0]>0.5)
	t_class = (t_preds==t_labels)
	t_data = t_data[:,2:]

	X_LENGTH = 50

	mean_s = np.mean(s_data, axis=0)[:X_LENGTH]
	mean_t_0 = np.mean(t_data[t_class==0,:], axis=0)[:X_LENGTH]
	mean_t_1 = np.mean(t_data[t_class==1,:], axis=0)[:X_LENGTH]

	x = np.linspace(0, X_LENGTH, X_LENGTH)

	fontsize = 18
	bar_width = 0.2

	fig = plt.figure(figsize=(18,6))
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.rcParams.update({'font.size': fontsize})
	plt.bar(x-bar_width,mean_s,width=bar_width,label=r"center(${\bf m_{o}}$) of source samples",color=colors[0],hatch='o',linewidth=0)
	plt.bar(x+bar_width,mean_t_0,width=bar_width,label=r"center(${\bf m_{o}}$) of false target samples",color=colors[1],hatch='+',linewidth=0)
	plt.bar(x,mean_t_1,width=bar_width,label=r"center(${\bf m_{o}}$) of true target samples",color=colors[2],hatch='*',linewidth=0)
	plt.xlabel("Dimensions", fontsize=fontsize)
	plt.ylabel("Value", fontsize=fontsize)
	plt.xticks(fontsize = fontsize) 
	plt.yticks(fontsize = fontsize) 
	plt.xlim([0,20])
	plt.ylim([-0.05,0.15])
	plt.title(r"Comparison of center(${\bf m_{o}}$) between true and false samples",fontsize=fontsize)
	plt.legend(loc='upper left', prop={'size': fontsize})
	plt.show()

	pp = PdfPages('mmd.pdf')
	pp.savefig(fig)
	pp.close()
	plt.close()

	os.system("pdfcrop mmd.pdf mmd.pdf")

def plot_bar_mmd2():
	t_data = read_matrix('mmd.txt')
	t_data = t_data[:,2:]

	X_LENGTH = 50

	mean_t_i = np.mean(t_data[0:300,:], axis=0)[:X_LENGTH]
	mean_t_j = np.mean(t_data[301:600,:], axis=0)[:X_LENGTH]

	x = np.linspace(0, X_LENGTH, X_LENGTH)

	fontsize = 18
	bar_width = 0.3

	fig = plt.figure(figsize=(18,6))
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.rcParams.update({'font.size': fontsize})
	plt.bar(x-bar_width,mean_t_i, width=bar_width, label=r"center(${\bf m_{o}}$) of batch i",color=colors[2],hatch='o',linewidth=0)
	plt.bar(x,mean_t_j,width=bar_width, label=r"center(${\bf m_{o}}$) of batch j",color=colors[0],hatch='+',linewidth=0)
	plt.xlabel("Dimensions", fontsize=fontsize)
	plt.ylabel("Value", fontsize=fontsize)
	plt.xticks(fontsize = fontsize) 
	plt.yticks(fontsize = fontsize) 
	plt.xlim([0,20])
	plt.title(r"Comparison of center(${\bf m_{o}}$) between batch i and j",fontsize=fontsize)
	plt.legend(loc='upper right', prop={'size': fontsize})
	plt.show()

	pp = PdfPages('mmd2.pdf')
	pp.savefig(fig)
	pp.close()
	plt.close()
	os.system("pdfcrop mmd2.pdf mmd2.pdf")

def plot_pr_curve():
	pr0 = read_matrix('pr0.txt')
	pr1 = read_matrix('pr1.txt')[4:,:]
	pr2 = read_matrix('pr2.txt')[4:,:]
	pr3 = read_matrix('pr3.txt')[3:,:]
	pr4 = read_matrix('pr4.txt')

	fontsize = 14

	fig = plt.figure(figsize=(8,6))
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.rcParams.update({'font.size': fontsize})
	plt.plot(pr0[:,1],pr0[:,2],label=r"$L_{S}({\bf X}^{S})$",marker='*',linewidth=2)
	plt.plot(pr1[:,1],pr1[:,2],label=r"$L_{S}({\bf X}^{T,n})$",marker='D',linewidth=2)
	plt.plot(pr2[:,1],pr2[:,2],label=r"$L_{S}({\bf X}^{T,n})+L_{EWM}$",marker='o',linewidth=2)
	plt.plot(pr3[:,1],pr3[:,2],label=r"$L_{S}({\bf X}^{T,n},{\bf X}^{S,n})+L_{FV}$",marker='v',linewidth=2)
	plt.plot(pr4[:,1],pr4[:,2],label=r"$L_{S}({\bf X}^{T,n},{\bf X}^{S,n})+L_{EWM}$",marker='s',linewidth=2)
	# plt.plot([0, 1], [1, 0], color='grey', linestyle='-', linewidth=1)
	plt.xlabel("1 - Precision", fontsize=fontsize)
	plt.ylabel("Recall", fontsize=fontsize)
	plt.xticks(fontsize = fontsize) 
	plt.yticks(fontsize = fontsize) 
	plt.xlim([0,0.45])
	# plt.ylim([0,1])
	plt.title("Precision-recall curve on target scene 1",fontsize = fontsize) 
	plt.legend(loc='lower right', prop={'size': fontsize})
	plt.show()

	pp = PdfPages('pr_curve.pdf')
	pp.savefig(fig)
	pp.close()
	plt.close()
	os.system("pdfcrop pr_curve.pdf pr_curve.pdf")

def plot_f1_score():
	f1 = read_matrix('f1.txt')
	f2 = read_matrix('f2.txt')
	f3 = read_matrix('f3.txt')[:15]
	f4 = read_matrix('f4.txt')[:15]

	fontsize = 14

	fig = plt.figure(figsize=(8,5))
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')
	plt.rcParams.update({'font.size': fontsize})
	plt.plot(f1,label=r"$L_{S}({\bf X}^{T,n})$",marker='D',linewidth=2)
	plt.plot(f2,label=r"$L_{S}({\bf X}^{T,n})+L_{EWM}$",marker='o',linewidth=2)
	plt.plot(f3,label=r"$L_{S}({\bf X}^{T,n},{\bf X}^{S,n})+L_{FV}$",marker='v',linewidth=2)
	plt.plot(f4,label=r"$L_{S}({\bf X}^{T,n},{\bf X}^{S,n})+L_{EWM}$",marker='s',linewidth=2)
	plt.xlabel("Adaptation iteration number", fontsize=fontsize)
	plt.ylabel("F1 Score", fontsize=fontsize)
	plt.xticks(fontsize = fontsize) 
	plt.yticks(fontsize = fontsize) 
	plt.xlim([0,14])
	plt.title("F1 Score changes during adaptation on target scene 1",fontsize = fontsize) 
	plt.legend(loc='lower right', prop={'size': fontsize})
	plt.show()

	pp = PdfPages('f1_score.pdf')
	pp.savefig(fig)
	pp.close()
	plt.close()
	os.system("pdfcrop f1_score.pdf f1_score.pdf")

if __name__ == "__main__":
	plot_bar_mmd()
	plot_bar_mmd2()
    	plot_pr_curve()
    	plot_f1_score()
