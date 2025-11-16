import PySimpleGUI as sg
import numpy as np
import cv2
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

def getImageData(cvimage):
    _, im_arr = cv2.imencode('.png', cvimage)
    im_bytes = im_arr.tobytes()
    base = base64.b64encode(im_bytes)
    return base

def name(name):
        NAME_SIZE = 23
        dots = NAME_SIZE-len(name)-2
        return sg.Text(name + ' ' + '•'*dots, size=(NAME_SIZE,1), justification='r',pad=(0,0), font='Courier 10')

# performs desired preprocessing on the image
def preprocess(image, type='Closing', ksize=7):
    kernel = np.ones((ksize,ksize))
    if type == 'Dilation':
        return cv2.dilate(image, kernel)
    elif type == 'Opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif type == 'Gaussian Blur':
        return cv2.GaussianBlur(image, (ksize,ksize), 0)
    elif type == 'Closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        return image

# performs automatic edge detection on image 
def edgeDetect(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0,(1.0-sigma)*v))
    upper = int(min(255,(1.0-sigma)*v))
    return cv2.Canny(image, lower, upper)

# denoises contours
def denoise(contours, thresh=10):
    return [i for i in contours if cv2.contourArea(i) >= thresh]
    # return [i for i in contours if len(i) >= thresh]

# combines all contours within thresh pixels or less - UNUSED
def combineContours(contours, thresh=10):
    'get average position for each contour for comparison'
    print('calculating contour averages')
    averages = np.zeros((len(contours),2))
    for i in range(len(contours)):
        xsum = 0; ysum = 0
        row = contours[i].shape[0]
        cnt = np.reshape(contours[i], (row, 2))
        for j in range(row):
            xsum += cnt[j][0]
            ysum += cnt[j][1]

        xsum /= row; ysum /= row
        averages[i] = [xsum, ysum]
        # print('contour =', contours[i], ', average is', averages[i])
    # np.savetxt('averages.csv', averages, delimiter=',')
    # print(averages)
    length = len(contours)
    status = np.zeros((length, 1))

    print('checking possible contour combination')
    # unified = []
    for i, cnt1 in enumerate(averages):
        x = i
        if i != length - 1:
            for j, cnt2 in enumerate(averages[i+1:]):
                x = x+1
                # dist = find_if_close(cnt1, cnt2)
                dist = abs(np.linalg.norm(cnt1-cnt2)) <= thresh
                if dist == True:
                    val = min(status[i], status[x])
                    # print('combining', cnt1, 'and', cnt2)
                    status[x] = status[i] = val
                    # unified.append
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1
    print('combining contours')
    unified = []
    maximum =  int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(cont)
    
    return unified

# CURRENTLY: gets average point of contours to compare distance
# MODIFY: rectangle bounds (no rotating) and checks shortest x or y offset 
def calcContourDistance(cnt1, cnt2):
    # xsum = 0; ysum = 0
    row1 = cnt1.shape[0]
    cnt1 = np.reshape(cnt1, (row1, 2))
    avg1 = np.average(cnt1, axis=0)
    # xsum = 0; ysum = 0
    row2 = cnt2.shape[0]
    cnt2 = np.reshape(cnt2, (row2, 2))
    avg2 = np.average(cnt2, axis=0)

    dist = abs(np.linalg.norm(avg1-avg2))
    # _, dot1 = getContourLength(cnt1)
    # _, dot2 = getContourLength(cnt2)

    # dist = min(abs(np.linalg.norm(dot1[0]-dot2[0])), 
    #            abs(np.linalg.norm(dot1[0]-dot2[1])), 
    #            abs(np.linalg.norm(dot1[1]-dot2[0])),
    #            abs(np.linalg.norm(dot1[1]-dot2[1])))
    
    return dist

# checks along line through both contours
def checkIfAlongLine(cnt1, cnt2):
    # init: check distance
    # if calcContourDistance(cnt1, cnt2) > threshold:
    #     return False
    # build lines for both contours
    length1, dots1 = getContourLength(cnt1)
    [vx1, vy1, x1, y1] = cv2.fitLine(np.asarray(dots1), cv2.DIST_L2, 0, 0.01, 0.01)
    # print('line is along', vx1, vy1)
    direction = abs(np.asarray([vx1[0], vy1[0]])); point = np.asarray([x1[0], y1[0]])
    for i in range(len(cnt2)):
        # print('test point is', cnt2[i][0])
        vector = abs(cnt2[i][0] - point)
        # print(vector, direction)
        # print('\n', np.dot(vector, direction), np.linalg.norm(vector), np.linalg.norm(direction))
        theta = np.arccos(np.dot(vector, direction)/(np.linalg.norm(vector)*np.linalg.norm(direction)))
        # print('theta is',theta)
        if theta < np.pi/6:
            return True
        
    length2, dots2 = getContourLength(cnt2)
    [vx2, vy2, x2, y2] = cv2.fitLine(np.asarray(dots2), cv2.DIST_L2, 0, 0.01, 0.01)
    direction = abs(np.asarray([vx2[0], vy2[0]])); point = np.asarray([x2[0], y2[0]])
    for i in range(len(cnt1)):
        vector = abs(cnt1[i][0] - point)
        theta = np.arccos(np.dot(vector, direction)/(np.linalg.norm(vector)*np.linalg.norm(direction)))
        if theta < np.pi/6:
            return True

    return False

# get average point in a contour
def getAvg(contour):
    tempcnt = np.reshape(contour, (contour.shape[0],2))
    return np.asarray(np.average(tempcnt, axis=0), dtype=int)

# do aggregate clustering for contour combination
def clusterContours(contours, threshold=20, mtype='Average'):
    currentCnts = contours
    # run until all contours are merged, or no contours to combine within threshold dist
    while len(currentCnts) > 1:
        if mtype == 'Average':
            pts = np.zeros((len(currentCnts),2), dtype=int)
            for i in range(len(currentCnts)):
                tempcnt = np.reshape(currentCnts[i], (currentCnts[i].shape[0],2))
                pts[i] = np.average(tempcnt, axis=0)
        elif mtype == 'Endpoint':
            pts = np.zeros((len(currentCnts),2,2), dtype=int)
            for i in range(len(currentCnts)):
                _, pts[i] = getContourLength(currentCnts[i])
        mindist = None
        mincoord = None
        # iterate through all contour combinations
        for a in range(len(currentCnts)-1):
            for b in range(a+1,len(currentCnts)):
                # # get distance between them
                # dist = calcContourDistance(currentCnts[a], currentCnts[b])
                if mtype == 'Average':
                    dist = abs(np.linalg.norm(pts[a]-pts[b]))
                elif mtype == 'Endpoint':
                    dist = min(abs(np.linalg.norm(pts[a,0]-pts[b,0])), 
                               abs(np.linalg.norm(pts[a,0]-pts[b,1])), 
                               abs(np.linalg.norm(pts[a,1]-pts[b,0])),
                               abs(np.linalg.norm(pts[a,1]-pts[b,1])))
                # hold minimum distance
                if mindist is None or mindist > dist:
                    mindist = dist
                    mincoord = (a,b)
        if mindist < threshold:
            # merge contours
            dex1, dex2 = mincoord
            currentCnts[dex1] = np.concatenate((currentCnts[dex1], currentCnts[dex2]), axis=0)
            del currentCnts[dex2]
        else:
            # there are no other contours within the threshold to combine
            break
    return currentCnts

# conversion of pixel distance to expected energy in keV
def pixToEnergy(x):
    # m = 2.03187251; b = -20.41832669
    ratio = int(values['scale']) / base.shape[1]
    nms = ratio * x
    # print('Lengths are (nm):', nms)
    # energy = nms * m + b

    # exponential fit
    a = 1.0048; b = -1549.68; c = -1591.95; d = 0.20412
    energy = a**(d*nms-b)+c

    return energy

# calculates the longest straight-line distance through a contour
def getContourLength(cnt):
    points = cnt.shape[0]
    # find average position of contour
    avg = np.average(cnt)
    # print('average is:', avg)
    std = np.std(cnt)

    longest = 0
    dots = np.zeros((2,2), np.uint8)
    i = 0
    for pt1 in range(points-1):
        for pt2 in range(pt1+1,points):
            if np.linalg.norm(cnt[pt2]-avg) > 3*std:
                continue
            dist = np.linalg.norm(cnt[pt1]-cnt[pt2])
            if abs(dist) > longest:
                longest = dist  
                dots = (cnt[pt1,0], cnt[pt2,0])
            i += 1

    # return lengths[longest], pixline
    return longest, dots

# calls getContourLength for each contour, compiles the results in a saveable way
def extractDistances(contours):
    lengthList = np.zeros((len(contours)))
    linelist = np.zeros((len(contours),2,2))
    for i in range(len(contours)):
        cnt = contours[i]
        # print(np.asarray(cnt))
        lengthList[i], linelist[i] = getContourLength(cnt)
    return lengthList, linelist

# rounds contours into convexHull for plotting
def fitContours(contours):
    return [cv2.convexHull(cnt) for cnt in contours]

    
'TO DO:'
# combine along a line fitted to the contour rather than centre point distances for calcContourDistances - ABANDONED
# use matchShapes to get categories of similar tracks - DONE
# sorting of contours by area, for both categorizing and saving - DONE
# implement saving of contours into some sort of data file
# denoise using the area of the contour rather than number of points - DONE
    # however much less aggressive
# Vignette processing using super blurred version of image?


if __name__ == '__main__':
    plt.rcParams.update({'font.size':14})
    base = None; img = None; trackHist = None; clusterHist = None
    # img = sg.popup_get_file('Choose file',default_path='./')
    # print(img)
    runtime = 0
    plotHist = True

    clusterGraph = None; trackGraph = None

    image_elem = sg.Image(filename=img)
    name = sg.Text(img, size=(80,3))

    col = [[sg.Text('Choose Image:'), sg.Input(key='-IN2-',change_submits=True,s=50), sg.FileBrowse(key='file'), sg.Button('Submit'), sg.Button('Reset')],
           [sg.Text('Image size: 0 x 0',key='pixsize')],
           [sg.Image(data=img, key='original'), sg.Image(data=img, key='img'), sg.Canvas(key='trackHist'), sg.Canvas(key='clusterHist')]]
    tlinput = [[sg.Text('Enter nm width:'), sg.Input(key='scale',s=10,default_text='260')],
               [sg.Button('Calculate Track Lengths')]]
    comparing = [[sg.Text('Similarity Threshold:'), sg.Input(key='matchThresh',s=5,default_text='0.5')],
                 [sg.Button('Compare Shapes')]]
    saving = [[sg.Text('Save Results:')],
              [sg.Input(key='saveto',change_submits=True, s=50), sg.FolderBrowse(key='savefolder'), sg.Button('Save')]]
    preproc = [[sg.Text('1.'), sg.Button('Preprocess')],
               [sg.Text('Type'), sg.Combo(['Closing', 'Gaussian Blur'], default_value='Closing', readonly=True, k='process')],
               [sg.Text('Kernel Size'), sg.Input(default_text='7', s=10,key='ksize')]]
    edge = [[sg.Text('2.'), sg.Button('Detect Edges')],
            [sg.Text('Sigma'), sg.Input(default_text='0.7', s=10,key='sigma')]]
    cnt = [[sg.Text('3.'), sg.Button('Find Contours')]]
    dnSlot = [[sg.Text('4.'), sg.Button('Denoise')],
            [sg.Text('Min Area'), sg.Input(default_text='2', s=10,key='denoise')]]
    comb = [[sg.Text('5.'), sg.Button('Combine Contours')],
            [sg.Text('Max Distance'), sg.Input(default_text='10', s=10,key='comb')],
            [sg.Text('Merge Type:'), sg.Combo(['Average', 'Endpoint'], default_value='Endpoint', readonly=True, k='mtype')]]
    fit = [[sg.Button('Fit Contours')]]

    layout = [[sg.Column(col)],
              [sg.Column(preproc), sg.VerticalSeparator(), sg.Column(edge), sg.VerticalSeparator(), sg.Column(cnt), sg.VerticalSeparator(), sg.Column(dnSlot), sg.VerticalSeparator(), sg.Column(comb)],
              [sg.HorizontalSeparator()],
              [sg.Column(tlinput), sg.VerticalSeparator(), sg.Column(comparing), sg.VerticalSeparator(), sg.Column(saving)]]
    window = sg.Window('Track Detector', layout, finalize=True,resizable=True, default_element_size=(40,1))

    window.bind('<Configure>',"Event")

    while True:
        event, values = window.Read()
        if event == sg.WIN_CLOSED:
            break

        # if event == "Event":


        if event == 'Submit':
            base = cv2.imread(values['file'])
            # base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            dims = base.shape
            # if base.shape[1] > 500:
            #     base = cv2.resize(base, [int(0.8*dims[0]),int(0.8*dims[1])])
            window['original'].update(getImageData(base))
            # window['filename'].update(values['file'])
            window['pixsize'].update(f'Image size: {dims[0]} x {dims[1]}')

        if event == 'Reset':
            runtime = 0
            img = np.copy(base)
            contours = None
            window['img'].update(getImageData(img))

        if event == 'Preprocess':
            start = time.time()
            img = np.copy(base)
            img = preprocess(img, type=values['process'], ksize=int(values['ksize']))
            runtime += time.time() - start
            window['img'].update(getImageData(img))

        if event == 'Detect Edges':
            start = time.time()
            img = edgeDetect(img, sigma=float(values['sigma']))
            runtime += time.time() - start
            window['img'].update(getImageData(img))

        if event == 'Find Contours':
            start = time.time()
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = fitContours(contours)
            runtime += time.time() - start
            img = np.copy(base)
            cv2.drawContours(img, contours, -1, (0,255,0), 1)
            window['img'].update(getImageData(img))

        if event == 'Denoise':
            start = time.time()
            contours = denoise(contours, thresh=float(values['denoise']))
            contours = fitContours(contours)
            runtime += time.time() - start
            img = np.copy(base)
            cv2.drawContours(img, contours, -1, (0,255,0), 1)
            window['img'].update(getImageData(img))

        if event == 'Combine Contours':
            # contours = combineContours(contours, thresh=float(values['comb']))
            start = time.time()
            contours = clusterContours(contours, threshold=float(values['comb']), mtype=values['mtype'])
            contours = fitContours(contours)
            runtime += time.time() - start
            img = np.copy(base)
            cv2.drawContours(img, contours, -1, (0,255,0), 1)
            window['img'].update(getImageData(img))

        # if event == 'Fit Contours':
        #     contours = [cv2.convexHull(cnt) for cnt in contours]
        #     img = np.copy(base)
        #     cv2.drawContours(img, contours, -1, (0,255,0), 1)
        #     window['img'].update(getImageData(img))
        #     # print('Contour:', contours[0])
        #     # print('Shape:', shape[0])

        if event == 'Calculate Track Lengths':
            start = time.time()
            # sorts contours largest to smallest
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            scale = values['scale']
            lengths, lines = extractDistances(contours)
            energies = pixToEnergy(lengths)
            print('Total energy =', sum(energies))
            # print('Energies are (keV):', energies)
            # print('lines:', lines)
            # img = np.copy(base)
            for i in range(len(lines)):
                pt1 = np.array(lines[i,0], dtype=int)
                pt2 = np.array(lines[i,1], dtype=int)
                cv2.line(img, pt1, pt2, [255,255,255], 1)
                cv2.putText(img, '{:.0f}'.format(energies[i]), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            runtime += time.time() - start
            window['img'].update(getImageData(img))
            # plt.hist(energies, rwidth=0.8)
            # plt.xlabel('Track Energy (keV)')
            # plt.ylabel('Count')
            # plt.show()
            if plotHist:
                fig, ax = plt.subplots(figsize=(4,3))
                ax.hist(energies, rwidth=0.8)
                ax.set_xlabel('Track Energy (keV)')
                ax.set_ylabel('Counts')
                fig.tight_layout()
                figcan = FigureCanvasTkAgg(fig, window['trackHist'].TKCanvas)
                figcan.draw()
                figcan.get_tk_widget().pack(side='top', fill='both', expand=True)
                dims = base.shape
                trackGraph = fig
            # window['trackHist'].update(size=(dims[0], dims[1]))

        if event == 'Compare Shapes':
            start = time.time()
            threshold = float(values['matchThresh'])
            idnums = np.zeros(len(contours), dtype=int)
            clusters = [[contours[0]]]
            clusterCount = [1]
            for i in range(1,len(contours)):
                # print('contour', i)
                bestMatch = 100
                bestid = -1
                for c in range(len(clusters)):
                    avg = 0
                    for ci in range(len(clusters[c])):
                        avg += cv2.matchShapes(contours[i], clusters[c][ci], 1, 0.0)
                    avg /= len(clusters[c])
                    if avg < bestMatch:
                        bestMatch = avg
                        bestid = c
                    # print('cluster', c, 'got', avg)
                if bestMatch < threshold:
                    idnums[i] = bestid
                    clusterCount[bestid] += 1
                    clusters[bestid].append(contours[i])
                else:
                    clusters.append([contours[i]])
                    idnums[i] = len(clusters)-1
                    clusterCount = np.append(clusterCount, 1)

            runtime += time.time() - start
            print('Runtime = ', runtime)
            print('Number of clusters:', len(clusters))
            print('Counts:', clusterCount)
            # print('idnums:', idnums)

            img = np.copy(base)
            drawing = []

            palette = sns.color_palette('husl', len(clusters))
            palette2 = palette.copy()
            for clr in range(len(palette2)):
                palette2[clr] = tuple([int(255*x) for x in palette2[clr]])
            for c in range(len(contours)):
                clr = tuple([255*x for x in palette2[idnums[c]]])
                cv2.drawContours(img, [contours[c]], 0, palette2[idnums[c]], 1)
                cv2.putText(img, str(idnums[c]), getAvg(contours[c]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
            window['img'].update(getImageData(img))
            # print(cv2.matchShapes(contours[0], contours[1], 1, 0.0))
            # cv2.drawContours(img, [contours[0], contours[1]], -1, (255,0,0), 1)
            # window['img'].update(getImageData(img))

            if plotHist:
                fig, ax = plt.subplots(figsize=(4,3))
                N, bins, patches = ax.hist(idnums, bins=len(clusters), rwidth=0.8)
                # print(len(patches), len(clusters))
                # for i in range(len(patches)):
                #     patches[i].set_facecolor(palette[i])
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Counts')
                ax.set_xticks([i for i in range(len(clusterCount))])
                fig.tight_layout()
                figcan = FigureCanvasTkAgg(fig, window['clusterHist'].TKCanvas)
                figcan.draw()
                figcan.get_tk_widget().pack(side='top', fill='both', expand=True)
                dims = base.shape

                clusterGraph = fig

        if event == 'Save':
            cv2.imwrite(values['savefolder']+'/resultImage.png', img)
            # print('Save completed at', values['savefolder']+'/resultImage.png')
            # print(type(contours))

            # window['trackHist'].save_element_to_disk(values['savefolder']+'energies.png')
            # window['clusterHist'].save_element_to_disk(values['savefolder']+'clusters.png')
            lengths = extractDistances(contours)
            # print('length =', lengths)
            if trackGraph != None:
                trackGraph.savefig(values['savefolder']+'/energies.png')
            if clusterGraph != None:
                clusterGraph.savefig(values['savefolder']+'/clusters.png')



