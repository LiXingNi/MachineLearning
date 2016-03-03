import matplotlib.pyplot as plt
from functools import reduce

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题



decision_node = dict(boxstyle='sawtooth',fc = "0.8")
leaf_node = dict(boxstyle = 'round4',fc = '0.8')
arrow_args = dict(arrowstyle="<-")

def getNumLeafs(tree_node):
    if len(tree_node.children_dic) == 0:
        return 1
    return reduce(lambda x,y:x+y,map(getNumLeafs,tree_node.children_dic.values()))



def getTreeDepth(tree_node):
    if len(tree_node.children_dic) == 0:
        return 1
    return reduce(lambda x,y : max(x,y),map(lambda x:1+getTreeDepth(x),tree_node.children_dic.values()))


##############################################################

def plotNode(node_txt,center_pt,parent_pt,node_type):
    createPlot.ax1.annotate(node_txt,xy = parent_pt,\
                               xycoords = 'axes fraction',\
                               xytext = center_pt,textcoords ='axes fraction',\
                               va = 'center',ha = 'center',bbox = node_type,\
                           arrowprops = arrow_args, size = 20)


def  plotMidText(cnt_pt,parent_pt,txt_string):
    x_mid =(parent_pt[0] - cnt_pt[0]) / 2.0 + cnt_pt[0]
    y_mid = (parent_pt[1] - cnt_pt[1]) / 2.0 + cnt_pt[1]
    createPlot.ax1.text(x_mid,y_mid,txt_string)


def plotTree(my_tree,parent_pt,node_txt):
    num_leafs = getNumLeafs(my_tree)
    depth = getTreeDepth(my_tree)
    first_attr = my_tree.feature
    cntr_pt = (plotTree.xOff + (1.0 + num_leafs) / 2 / plotTree.totalW, plotTree.yOff )
    plotMidText(cntr_pt, parent_pt, node_txt)
    plotNode(first_attr, cntr_pt, parent_pt, decision_node)
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    for key, item in my_tree.children_dic.items():
        if len(item.children_dic) != 0:
            plotTree(item, cntr_pt, key)
        else:
            plotTree.xOff = plotTree.xOff + 1 / plotTree.totalW
            plotNode(item.feature, (plotTree.xOff, plotTree.yOff), cntr_pt, leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), cntr_pt, key)
    plotTree.yOff = plotTree.yOff + 1 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)
    plotTree.totalW = getNumLeafs(inTree)
    plotTree.totalD = getTreeDepth(inTree)
    plotTree.xOff = - 0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
