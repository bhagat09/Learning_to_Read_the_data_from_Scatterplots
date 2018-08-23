import matplotlib
matplotlib.use('Agg') 
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox #matplotlib function for the bounding boxes
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm
from matplotlib import font_manager
import random
import os
import argparse
import scatteract_logger

import pandas as pd

#c=[1,2,3,4,5,6,7,1,2,3,4,5,6,3,5,3,4,5,6,4,3,1,2,13,14,23,45,67,89,78,67,5,43,23,4,5,66,77,88,99,33,22,11]
#d=[11,2,3,4,6,5,3,4,6,3,22,44,22,44,55,67,88,66,44,22,4,6,77,5,44,33,5,6,7,55,33,44,55,66,11,221]




# different types of parameters in matplotlib
# https://matplotlib.org/api/pyplot_api.html

markers = np.array([".",",","o","v","^","<",">","1","2","3","4","8","s","p"])
diff_markers = [markers[j] for j in [0,1,2,3,4,5,6,11,12,13]]
color_subtick_list = ['b','g','r','k', 'k', 'k', '0.7', '0.85']


tickDirection = ['inout','in','out']

#fonts from the a list containing different types of fonts
list_font = ['fonts/' + name for name in os.listdir('fonts')]
 
#dots per inch (resolution)
minimum_dpi = 85 #minimum resolution
maximum_dpi = 250 #maximum resolution


#defining the size of the ticks 
tick_size_width_min = 0
tick_size_width_max = 3
tick_size_length_min = 0
tick_size_length_max = 12

#size of the points 
minPointSize = 3
maxPointSize = 12



#defining the minimum and maximum size of the figures 
figsize_min = 3
figsize_max = 10

max_points_variations = 5

#padding between the tick marks and the tick values
pad_min = 2
pad_max = 18

#size of the axes labels
SizeAxisLabelMin = 8
SizeAxisLabelMax = 14
SizeTickLabelMin = 8
SizeTickLabelMax = 14
SizeTitleMin = 11
SizeTitleMax = 20

#length of the axis labels
LengthAxisLabelMin = 5
LengthAxisLabelMax = 12
LengthTitleMin = 5
LengthTitleMax = 20

colorbg_transparant_max = 0.05


point_dist = ['uniform', 'linear', 'quadratic']

#styling in scatterplots
styles = plt.style.available

#removing the backgrounds which are dark so that the points are properly visible 
if 'dark_background' in styles:
    styles.remove('dark_background')


#defining a function for the category of the points
def category(cat,cat_dict):

    for key, cat_i in cat_dict.items():

        if cat[0]==cat_i[0] and np.all(cat[1]==cat_i[1]) and cat[2]==cat_i[2] and cat[3]==cat_i[3]:
            return key

    return False


def scatter_plot(name, direc):
    """
    This is a function which will generate the scatterplots with lots of variance
    Parameters:
    nameof_plot:name of the plot which will be saved.
    direc:directory
    Returns:
    ax : Axis of the plots
    fig :Figure of the plot
    x, y : The X and Y coordinates of the plots
    s : Size of the points
    categories_of_points : Categories of the points
    tick_size : Tick size on the plot
    xAxisPos, yAxisPos: Position of the labels of the axis.
    
    """
    
    #using different stydes for plotting in matplotlib
    style = random.choice(styles)
    plt.style.use(style)
    
   


    #dpi and size of the ticks
    dpi = int(minimum_dpi + np.random.rand(1)[0]*(maximum_dpi-minimum_dpi)) #randomly selecting dpi with predefined maximum dpi and minimum dpi values
    figsize = (figsize_min+np.random.rand(2)*(figsize_max-figsize_min)).astype(int) #randomly selecting the size of the figures
    tick_size = [(tick_size_width_min+np.random.rand(1)[0]*(tick_size_width_max-tick_size_width_min)),
                 (tick_size_length_min+np.random.rand(1)[0]*(tick_size_length_max-tick_size_length_min))]
    tick_size.sort()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    #ax=plt.axes()
    
    #using pandas to read the dataset (Bike Sharing Dataset)
    df=pd.read_csv("hour.csv")
    dataframe=pd.DataFrame(df)
    c=dataframe.iloc[:,12] # reading the humidity column in the dataset 
    d=dataframe.iloc[:,16] # reading the count column in the dataset
    x=random.sample(c,4) #taking 20 random values from the c column as X axis
    y=random.sample(d,4) #taking 20 random values from the d column as Y axis

     # variance in points
    pointsVar = 1+int(np.random.rand(1)[0]*max_points_variations) #setting up some random variance values
    pointsColors =  1+int(np.random.rand(1)[0]*pointsVar) #setting up the color of the points based on the variance values
    pointsMarkers =  1+int(np.random.rand(1)[0]*(pointsVar-pointsColors)) #setting up the markers of the points based on the variance values and colors
    pointsSize =  max(1,1+pointsVar-pointsColors-pointsMarkers) #setting up the size of the points based on the variance values, colors and markers

    random_Color = np.random.rand(1)[0]
    if random_Color<=0.5:
        colors = cm.jet(np.random.rand(pointsColors)) #colormaps in matplotlib ## https://matplotlib.org/users/colormaps.html
    elif random_Color>0.5 and random_Color<=0.8:
        colors = cm.hot(np.random.rand(pointsColors))
    else:
        colors = cm.bone(np.linspace(0,0.6,pointsColors))
    s_set = (maxPointSize+np.random.rand(pointsSize)*(maxPointSize-minPointSize))**2
    markers_subset = list(np.random.choice(markers,size=pointsMarkers))
    markers_empty = np.random.rand(1)[0]>0.75
    markers_empty_ratio = random.choice([0.0,0.5])

        
    
   
    #generating the plots using X and Y values
    s = []
    categories_of_points = []
    cat_dict = {}
    index_cat = 0
    #zipped_data_x=[]
    #zipped_data_y=[]

    for i,j in zip(x,y):
        s_ = random.choice(s_set)
        c_ = random.choice(colors)
        m_ = random.choice(markers_subset)
        #zipped_data_x.append(chunked_data_x[i])
        #zipped_data_y.append(chunked_data_y[j])
        if m_ in diff_markers and markers_empty:
            e_ = np.random.rand(1)[0]> markers_empty_ratio
        else:
            e_ = False
        cat = [s_,c_,m_, e_]

        if category(cat,cat_dict) is False:
            cat_dict[index_cat] = cat
            index_cat += 1
        categories_of_points.append(category(cat,cat_dict))
        s.append(s_)
        if e_:
            plt.scatter(i,j, s=s_, color = c_, marker=m_, facecolors='none')
        else:
            plt.scatter(i,j, s=s_, color = c_, marker=m_)

    # padding between ticks and labels 
    padX = max(tick_size[1]+0.5,int(pad_min + np.random.rand(1)[0]*(pad_max-pad_min)))
    padY = max(tick_size[1]+0.5,int(pad_min + np.random.rand(1)[0]*(pad_max-pad_min)))
    directionX = random.choice(tickDirection)
    directionY = random.choice(tickDirection)

    # tick prob
    ticksProb = np.random.rand(1)[0]

    # style and location of the ticks for the X axis
    if np.random.rand(1)[0]>0.5:
        xAxisPos = 1
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        if ticksProb >0.8:
            ax.xaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=padX,
                                 direction= directionX, bottom=np.random.rand(1)[0]>0.5, top=True)
        else:
            ax.xaxis.set_tick_params(bottom=np.random.rand(1)[0]>0.5, top=True)
        if np.random.rand(1)[0]>0.5:
            ax.spines['bottom'].set_visible(False) #setting up the axes position
            ax.xaxis.set_tick_params(bottom=False)
            if np.random.rand(1)[0]>0.5:
                xAxisPos = np.random.rand(1)[0]
                ax.spines['top'].set_position(('axes',xAxisPos )) #setting up the axes (X) position
    else:
        xAxisPos = 0
        if ticksProb >0.8:
            ax.xaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=padX,
                                 direction= directionX, bottom=True, top=np.random.rand(1)[0]>0.5)
        else:
            ax.xaxis.set_tick_params(bottom=True, top=np.random.rand(1)[0]>0.5)
        if np.random.rand(1)[0]>0.5:
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_tick_params(top=False)
            if np.random.rand(1)[0]>0.5:
                xAxisPos = np.random.rand(1)[0]
                ax.spines['bottom'].set_position(('axes',xAxisPos))

    # style and location of the ticks for the Y axis
    if np.random.rand(1)[0]>0.5:
        yAxisPos = 1
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        if ticksProb > 0.8:
            ax.yaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=padY,
                                 direction= directionY, left=np.random.rand(1)[0]>0.5, right=True)
        else:
            ax.yaxis.set_tick_params(left=np.random.rand(1)[0]>0.5, right=True)
        if np.random.rand(1)[0]>0.5:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_tick_params(left=False)
            if np.random.rand(1)[0]>0.5:
                yAxisPos = np.random.rand(1)[0]
                ax.spines['right'].set_position(('axes',yAxisPos))
    else:
        yAxisPos = 0
        if ticksProb >0.8:
            ax.yaxis.set_tick_params(width=tick_size[0], length=tick_size[1], color='black', pad=padY,
                                 direction= directionY, left=True, right=np.random.rand(1)[0]>0.5)
        else:
            ax.yaxis.set_tick_params(left=True, right=np.random.rand(1)[0]>0.5)
        if np.random.rand(1)[0]>0.5:
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_tick_params(right=False)
            if np.random.rand(1)[0]>0.5:
                yAxisPos = np.random.rand(1)[0]
                ax.spines['left'].set_position(('axes',yAxisPos))
                
     # SUB-TICKs
    if ticksProb > 0.8:
        color_subtick = random.choice(color_subtick_list)
        length_subtick = 0.75*np.random.rand(1)[0]*tick_size[1]
        if np.random.rand(1)[0]>0.7:
            minorLocator = AutoMinorLocator()
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_tick_params(which='minor', length=length_subtick, direction= directionX, color=color_subtick,
                                     bottom= ax.spines['bottom'].get_visible(), top=ax.spines['top'].get_visible())
        if np.random.rand(1)[0]>0.7:
            minorLocator = AutoMinorLocator()
            ax.yaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_tick_params(which='minor', length=length_subtick, direction= directionY, color=color_subtick,
                                     left= ax.spines['left'].get_visible(), right= ax.spines['right'].get_visible())


    

    xmin = min(x)
    xmax = max(x)
    ignoreX = 0.05*abs(xmax-xmin)
    plt.xlim(xmin - ignoreX, xmax + ignoreX) # avoiding the points to get start from the axis
    ymin = min(y)
    ymax = max(y)
    ignoreY = 0.05*abs(ymax-ymin)
    plt.ylim(ymin - ignoreY, ymax + ignoreY)


    # font and size of the tick labels, axes labels and title
    font = random.choice(list_font)
    tickSize = int(SizeTickLabelMin + np.random.rand(1)[0]*(SizeTickLabelMax-SizeTickLabelMin))
    axesSize = int(SizeAxisLabelMin + np.random.rand(1)[0]*(SizeAxisLabelMax-SizeAxisLabelMin))
    titleSize = int(SizeTitleMin + np.random.rand(1)[0]*(SizeTitleMax-SizeTitleMin))
    ticks_font = font_manager.FontProperties(fname = font, style='normal', size=tickSize, weight='normal', stretch='normal')
    axes_font = font_manager.FontProperties(fname = font, style='normal', size=axesSize, weight='normal', stretch='normal')
    title_font = font_manager.FontProperties(fname = font, style='normal', size=titleSize, weight='normal', stretch='normal')

    x_label = "humidity"
    y_label = "count" 
    title = "Bike Sharing dataset"
    plt.xlabel(x_label , fontproperties = axes_font)
    plt.ylabel(y_label , fontproperties = axes_font, color='black')
    if xAxisPos==1:
        plt.title(title, fontproperties = title_font, color='black',y=1.1)
    else:
        plt.title(title, fontproperties = title_font, color='black')

    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    
    # background colors
    if np.random.rand(1)[0]>0.75:
        color_bg = (1-colorbg_transparant_max)+colorbg_transparant_max*np.random.rand(3)
        ax.set_axis_bgcolor(color_bg)
    if np.random.rand(1)[0]>0.75:
        color_bg = (1-colorbg_transparant_max)+colorbg_transparant_max*np.random.rand(3)
        fig.patch.set_facecolor(color_bg)

    plt.tight_layout()

    plt.savefig("./data/{}/".format(direc)+name, dpi='figure', facecolor=fig.get_facecolor())

    return ax, fig, x, y, s, categories_of_points, tick_size, xAxisPos, yAxisPos


def boundingBoxesPoints(ax, fig, x, y, s):
    
    # Method to get the bouding boxes of the points


    xy_pixels = ax.transData.transform(np.vstack([x,y]).T) #converting into the pixel coordinates
    xpix, ypix = xy_pixels.T

    boxes = []
    for xValue, yValue, siz in zip(xpix,ypix,s):
        if siz<25:
            siz = 25
        box_size = fig.dpi*np.sqrt(siz)/70.0
        x0 = xValue-box_size/2.0
        y0 = yValue - box_size/2.0
        x1 = xValue + box_size/2.0
        y1 = yValue + box_size/2.0
        boxes.append(Bbox([[x0, y0], [x1, y1]]))

    return boxes


def BoundingBoxTicks(ax, fig, tick_size, xAxisPos, yAxisPos):
    
    # Method that return the bouding box of the ticks.
    

    xTickPos = [ ax.transLimits.transform(textobj.get_position()) for textobj in ax.get_xticklabels() if len(textobj.get_text())>0]
    yTickPos = [ ax.transLimits.transform(textobj.get_position()) for textobj in ax.get_yticklabels() if len(textobj.get_text())>0]

    xTickPos = [ ax.transScale.transform(ax.transAxes.transform([array[0], xAxisPos])) for array in xTickPos]
    yTickPos = [ ax.transScale.transform(ax.transAxes.transform([yAxisPos, array[1]])) for array in yTickPos]

    box_x = []
    for x_list, y_list in xTickPos:
        sizeof_box_x = fig.dpi*5/50.0
        sizeof_box_y = fig.dpi*5/50.0 
        x0 = x_list-sizeof_box_x/2.0
        y0 = y_list-sizeof_box_y/2.0
        x1 = x_list+sizeof_box_x/2.0
        y1 = y_list+sizeof_box_y/2.0
        box_x.append(Bbox([[x0, y0], [x1, y1]]))

    box_y = []
    for x_list, y_list in yTickPos:
        sizeof_box_x = fig.dpi*5/50.0 
        sizeof_box_y = fig.dpi*5/50.0 
        x0 = x_list-sizeof_box_x/2.0
        y0 = y_list-sizeof_box_y/2.0
        x1 = x_list+sizeof_box_x/2.0
        y1 = y_list+sizeof_box_y/2.0
        box_y.append(Bbox([[x0, y0], [x1, y1]]))

    return box_x, box_y


def BoundingBoxLabels(ax):
    
    #Method that return the bouding box of the labels.
    
    boxLabel_x = [ textobj.get_window_extent() for textobj in ax.get_xticklabels() if len(textobj.get_text())>0]
    boxLabel_y = [ textobj.get_window_extent() for textobj in ax.get_yticklabels() if len(textobj.get_text())>0]

    return boxLabel_x, boxLabel_y


def LabelValues(ax):
    
    # Function that return the value of the labels.
    
    valuesOfLabel_x, valuesOfLabel_y = [], []

    xticksValues = ax.get_xticklabels()
    xticks_numbers = ax.get_xticks()
    for j in range(len(xticksValues)):
        if len(xticksValues[j].get_text())>0:
            valuesOfLabel_x.append(xticks_numbers[j])

    yticks_text = ax.get_yticklabels()
    yticks_numbers = ax.get_yticks()
    for j in range(len(yticks_text)):
        if len(yticks_text[j].get_text())>0:
            valuesOfLabel_y.append(yticks_numbers[j])

    return valuesOfLabel_x, valuesOfLabel_y


def returning_all(ax,fig, x, y, s, tick_size, xAxisPos, yAxisPos):
    
    #Method that return the bounding boxes and label values(required ones)
   
    boxesPoints = boundingBoxesPoints(ax, fig, x, y, s)

    tickBoxes_x, tickBoxes_y = BoundingBoxTicks(ax, fig, tick_size, xAxisPos, yAxisPos)

    labelBoxes_x, labelBoxes_y = BoundingBoxLabels(ax)

    valuesOfLabel_x, valuesOfLabel_y = LabelValues(ax)

    return boxesPoints, tickBoxes_x, tickBoxes_y, labelBoxes_x, labelBoxes_y, valuesOfLabel_x, valuesOfLabel_y



def coorToIdl(file_, plot_name, x , y):
    
    #method that writes the coordinate values of the points into an idl file
    
    name_plot = '"{plot_name}":'.format(plot_name=plot_name)
    for x_i, y_i in zip(x,y):
        name_plot += " ({}, {}),".format(x_i,y_i)
    name_plot = name_plot[:-1]
    name_plot+=';'
    file_.write(name_plot)
    file_.write("\n")


def formatIdl(length_y, file_, plot_name, boxes, scores = None):
    
    # format for the bounding boxes of the idl file
    
    name_plot = '"{plot_name}":'.format(plot_name=plot_name)
    if scores is None:
        for box in boxes:
            name_plot += " ({}, {}, {}, {}),".format(int(np.round(box.x0)),int(length_y-np.round(box.y1)),
                          int(np.round(box.x1)),int(length_y-np.round(box.y0)))
    else:
        for box, score in zip(boxes,scores):
            name_plot += " ({}, {}, {}, {}):{},".format(int(np.round(box.x0)),int(length_y-np.round(box.y1)),
                          int(np.round(box.x1)),int(length_y-np.round(box.y0)),score)
    name_plot = name_plot[:-1]
    name_plot+=';'
    file_.write(name_plot)
    file_.write("\n")


def labelsToIdl(length_y, _file, plot_name, labels, label_box):
    
    # method that writes label values into an idl _file.
    
    name_plot = '"{plot_name}":'.format(plot_name=plot_name)
    for j in range(len(labels)):
        box = label_box[j]
        name_plot += " ({}, {}, {}, {}):{},".format(int(np.round(box.x0)),int(length_y-np.round(box.y1)),
                          int(np.round(box.x1)),int(length_y-np.round(box.y0)),labels[j])
    name_plot = name_plot[:-1]
    name_plot+=';'
    _file.write(name_plot)
    _file.write("\n")


def FinalScatterplots(n, file_name, direc):
    
    #Function that combines all the previous funnctions and generate the plots and create all the idl file_s.
   

    if not os.path.exists("./data/{}".format(direc)):
        os.makedirs("./data/{}".format(direc))
        os.makedirs("./data/{}/plots".format(direc))
    
    with open("./data/{}/".format(direc)+file_name+"_coords.idl",'w') as f_coords, \
         open("./data/{}/".format(direc)+file_name+"_points.idl",'w') as f_points, \
         open("./data/{}/".format(direc)+file_name+"_points_cat.idl",'w') as f_points_cat, \
         open("./data/{}/".format(direc)+file_name+"_ticks.idl",'w') as f_ticks, \
         open("./data/{}/".format(direc)+file_name+"_labels.idl",'w') as f_labels, \
         open("./data/{}/".format(direc)+file_name+"_label_values.idl",'w') as f_label_values:

        for j in range(n):
            try:
                plot_name =  'plots/{}_{}.png'.format(file_name,j+1)
                ax, fig, x, y, s, categories_of_points, tick_size, xAxisPos, yAxisPos = scatter_plot(plot_name, direc)
                length_y = fig.get_size_inches()[1]*fig.dpi

                boxesPoints, tickBoxes_x, tickBoxes_y, labelBoxes_x, labelBoxes_y, valuesOfLabel_x, valuesOfLabel_y = returning_all(ax, fig, x, y, s, tick_size, xAxisPos, yAxisPos)
                coorToIdl(f_coords, plot_name, x , y)
                labelsToIdl(length_y, f_label_values, plot_name, valuesOfLabel_x+valuesOfLabel_y, labelBoxes_x+labelBoxes_y)
                formatIdl(length_y, f_points, plot_name, boxesPoints)
                formatIdl(length_y, f_points_cat, plot_name, boxesPoints, scores = categories_of_points)
                formatIdl(length_y, f_ticks, plot_name,  tickBoxes_x+tickBoxes_y)
                formatIdl(length_y, f_labels, plot_name, labelBoxes_x+labelBoxes_y)

                plt.close(fig)
            except ValueError:
                mylogger.warn("Error while generating plot.  This happens occasionally because of the tight-layout option.")


if __name__ == '__main__':

    """
    Example of command-line usage:

    python generate_random_scatter.py --directory plots_v1 --n_train 25000 --n_test 500
    """

    mylogger = scatteract_logger.get_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_train', help='Number of training images', required=True)
    parser.add_argument('--n_test', help='Number of test images', required=True)
    parser.add_argument('--directory', help='Directory to save the idl and images', required=True)
    args = vars(parser.parse_args())

    FinalScatterplots(n=int(args['n_train']), file_name = "train", direc = args['directory'])
    FinalScatterplots(n=int(args['n_test']), file_name = "test", direc = args['directory'])
