import numpy as np
import pyvista as pv # pip3 install pyvista
import matplotlib.pyplot as plt
import math
import copy
import sys
import os



################################
####    QUIC-Fire Input     ####
################################

class FbClass:
    def __init__(self):
        self.i = None
        self.j = None
        self.k = None
        self.state = None
        self.time = None


class ImgClass:
    def __init__(self):
        self.figure_size = None
        self.axis_font = None
        self.title_font = None
        self.colorbar_font = None


class LinIndexClass:
    def __init__(self):
        self.ijk = None
        self.num_cells = None


class FlagsClass:
    def __init__(self):
        self.firebrands = None
        self.en2atm = None
        self.perc_mass_burnt = None
        self.fuel_density = None
        self.emissions = None
        self.thermal_rad = None
        self.qf_winds = None
        self.qu_qwinds_inst = None
        self.qu_qwinds_ave = None
        self.react_rate = None
        self.moisture = None


class IgnitionClass:
    def __init__(self):
        self.hor_plane = None
        self.flag = None


class SimField:
    def __init__(self):
        self.nSensor = None
        self.nx = None
        self.ny = None
        self.nz = None
        self.dx = None
        self.dy = None
        self.dz = None
        self.z = None
        self.zm = None
        self.isfire = None
        self.sim_time = None
        self.ntimes = None
        self.time = None
        self.ntimes_ave = None
        self.time_ave = None
        self.dt = None
        self.dt_print = None
        self.dt_print_ave = None
        self.horizontal_extent = None
        self.indexing = LinIndexClass()





def import_inputs(plotParams):
    print("Importing input data")

    # Initialization
    qu = SimField()
    qf = SimField()
    ignitions = IgnitionClass()
    flags = FlagsClass()
    fb = FbClass()

    # Read input files
    read_qu_grid(qu,plotParams['PROJECTDIR'])
    read_qfire_file(qf, qu, ignitions, flags, fb,plotParams['PROJECTDIR'])

    return qu, qf, ignitions, flags, fb

def read_qu_grid(qu,projPath):

    # ------- QU_simparams
    fid = open_file(os.path.join(projPath,'QU_simparams.inp'), 'r')
    fid.readline()  # header
    qu.nx = get_line(fid, 1)
    qu.ny = get_line(fid, 1)
    qu.nz = get_line(fid, 1)
    qu.dx = get_line(fid, 2)
    qu.dy = get_line(fid, 2)
    grid_flag = get_line(fid, 1)
    if grid_flag == 0:
        temp = get_line(fid, 2)
        qu.dz = np.ones(qu.nz) * temp
    else:
        fid.readline()
        fid.readline()
        fid.readline()
        qu.dz = []
        for i in range(0, qu.nz):
            qu.dz.append(get_line(fid, 2))
    qu.nSensor = np.int(get_line(fid, 1))
    fid.close()

    read_vertical_grid(qu,projPath)

    qu.horizontal_extent = [0., qu.dx * float(qu.nx), 0., qu.dy * float(qu.ny)]

def read_ignitions(fid, qf, ignitions):
    fid.readline()  # ! IGNITION LOCATIONS
    ignitions.flag = get_line(fid, 1)

    # Specify 2D array of ignitions
    ignitions.hor_plane = np.zeros((qf.ny, qf.nx))

    if ignitions.flag == 1:  # line
        set_line_fire(fid, ignitions, qf)
        n = 1

    elif ignitions.flag == 2:  # square circle
        set_square_circle(fid, ignitions, qf)
        n = 1

    elif ignitions.flag == 3:  # circle
        set_circle(fid, ignitions, qf)
        n = 1

    elif ignitions.flag == 4:  # QF_Ignitions.inp
        n = 0

    elif ignitions.flag == 5:  # QF_IgnitionPattern.inp
        n = 0

    elif ignitions.flag == 6:  # ignite.dat
        set_firetech_ignitions(ignitions, qf)
        n = 1

    elif ignitions.flag == 7:  # ignite.dat
        #set_firetech_ignitions(ignitions, qf)
        n = 1

    else:
        n = 0

    # Read uninteresting lines
    for i in range(0, n):
        fid.readline()


def read_firebrands(fb,projPath):
    fname = projPath+'firebrands.bin'
    nelem = int(os.path.getsize(fname) / 4)
    fid = open_file(fname, 'rb')
    var = np.fromfile(fid, dtype=np.int32, count=nelem)
    var = np.reshape(var, ((5+2), int(nelem / (5+2))), order='F')
    fb.time = var[1]
    fb.i = var[2] - 1
    fb.j = var[3] - 1
    fb.k = var[4] - 1
    fb.state = var[5]


def read_file_flags(fid, flags):
    # Firebrands
    fid.readline()  # FIREBRANDS
    flags.firebrands = get_line(fid, 1)

    # Out files
    fid.readline()  # OUTPUT FILES
    flags.en2atm = get_line(fid, 1)
    flags.react_rate = get_line(fid, 1)
    flags.fuel_density = get_line(fid, 1)
    flags.qf_winds = get_line(fid, 1)
    flags.qu_qwinds_inst = get_line(fid, 1)
    flags.qu_qwinds_ave = get_line(fid, 1)
    fid.readline()
    flags.moisture = get_line(fid, 1)
    flags.perc_mass_burnt = get_line(fid, 1)
    fid.readline()
    flags.emissions = get_line(fid, 1)
    flags.thermal_rad = get_line(fid, 1)


def read_path(fid, qf):
    fid.readline()  # ! PATH LABEL
    temp = fid.readline()  # ! PATH
    qf.path = temp[1:-1]
    

def read_qfire_file(qf, qu, ignitions, flags, fb, projPath):
    fid = open_file(os.path.join(projPath,'QUIC_fire.inp'), 'r')

    read_times(fid, qu, qf)
    if(qu.isfire ==1):
        read_fire_grid(fid, qu, qf,projPath)
        read_path(fid, qf)
        read_fuel(fid)
        read_ignitions(fid, qf, ignitions)
        read_file_flags(fid, flags)
        if flags.firebrands == 1:
            read_firebrands(fb)
    read_topo_flags()

    fid.close()

def read_topo_flags():
    #nothing to do yet
    blah=0


def read_fire_grid_indexing(qf, projPath):
    global folderStr
    fid = open_file(os.path.join(projPath,folderStr,'fire_indexes.bin'), 'rb')
    np.fromfile(fid, dtype=np.int32, count=1)
    temp = np.fromfile(fid, dtype=np.int32, count=1)
    qf.indexing.num_cells = temp[0]
    np.fromfile(fid, dtype=np.int32, count=7 + qf.indexing.num_cells)
    qf.indexing.ijk = np.zeros((qf.indexing.num_cells, 3))
    for i in range(0, 3):
        qf.indexing.ijk[::1, i] = np.fromfile(fid, dtype=np.int32, count=qf.indexing.num_cells)

    qf.indexing.ijk = qf.indexing.ijk.astype(int)
    qf.indexing.ijk -= 1
    fid.close()

def read_times(fid, qu, qf):
    # Fire times
    qu.isfire = int(get_line(fid,1))
    for i in range(0, 3):
        fid.readline()

    # Simulation time
    qu.sim_time = int(get_line(fid, 1))
    qf.sim_time = qu.sim_time

    # Fire time step
    qf.dt = int(get_line(fid, 1))

    # QU time step
    qu.dt = int(get_line(fid, 1) * qf.dt)

    # Print time for FireCA variables
    qf.dt_print = int(get_line(fid, 1) * qf.dt)

    # Print time for QUIC variables
    qu.dt_print = int(get_line(fid, 1) * qu.dt)

    # Print time for emission and rad variables
    qf.dt_print_ave = int(get_line(fid, 1) * qf.dt)

    # Print time for averaged QUIC variables
    qu.dt_print_ave = int(get_line(fid, 1) * qu.dt)

    qf.ntimes = int(qf.sim_time / qf.dt_print + 1)
    qf.time = np.zeros((qf.ntimes,), dtype=np.int)

    qu.ntimes = int(qu.sim_time / qu.dt_print + 1)
    qu.time = np.zeros((qu.ntimes,), dtype=np.int)

    for i in range(0, qf.ntimes):
        qf.time[i] = qf.dt_print * i

    for i in range(0, qu.ntimes):
        qu.time[i] = qu.dt_print * i

    qf.ntimes_ave = int(qf.sim_time / qf.dt_print_ave + 1)
    qf.time_ave = np.zeros((qf.ntimes_ave,), dtype=np.int)

    qu.ntimes_ave = int(qu.sim_time / qu.dt_print_ave + 1)
    qu.time_ave = np.zeros((qu.ntimes_ave,), dtype=np.int)

    for i in range(0, qf.ntimes_ave):
        qf.time_ave[i] = qf.dt_print_ave * i

    for i in range(0, qu.ntimes_ave):
        qu.time_ave[i] = qu.dt_print_ave * i

def read_fireca_field(filestr, ntimes, times, qf, is_3d, plotParams, *args, **kwargs):
    global folderStr
    outvar = []
    print('Reading:',filestr,'Horiz dims:',qf.nx,qf.ny)
    if (filestr == "mburnt_integ-" or filestr =="h"):
        nvert = 1
    else:
        c = kwargs.get('c', None)
        layers = kwargs.get('layers', None)
        if(layers == None):
            nvert = qf.nz
        else:
            nvert = layers

    for i in range(0, ntimes):
        fname = os.path.join(plotParams['PROJECTDIR'],folderStr,filestr + '%05d.bin' % (times[i]))
        # Open file
        fid = open_file(fname, 'rb')
        temp = np.zeros((qf.ny, qf.nx, nvert))
        # Read header
        np.fromfile(fid, dtype=np.float32, count=1)
        if is_3d == 0:
            var = np.fromfile(fid, dtype=np.float32, count=qf.indexing.num_cells)[:]
            # http://scipy-cookbook.readthedocs.io/items/Indexing.html
            index = [qf.indexing.ijk[::, 1], qf.indexing.ijk[::, 0], qf.indexing.ijk[::, 2]]
            temp[index] = var
            #DRAWFIRE CHANGE
        else:
            temp = np.zeros((qf.ny, qf.nx, nvert))
            for k in range(0, nvert):
                t = np.fromfile(fid, dtype=np.float32, count=qf.nx * qf.ny)[:]
                temp[::1, ::1, k] = np.reshape(t, (qf.ny, qf.nx))
        outvar.append(temp)
        fid.close()
    outvar = np.moveaxis(np.asarray(outvar),3,1)

    return outvar

def read_wind_field(filestr, ntimes, times, qu, is_3d, *args, **kwargs):
    global folderStr
    outvar = []
    if (filestr == "mburnt_integ-" or filestr =="h"):
        nvert = 1
    else:
        c = kwargs.get('c', None)
        layers = kwargs.get('layers', None)
        if(layers == None):
            nvert = qu.nz
        else:
            nvert = layers


    print('Reading:',filestr,'Horiz dims:',qu.nx-1,qu.ny-1,' Number of Time Slices: ',ntimes)
    print('Number of vertical layers: ',nvert)
    for i in range( ntimes):
        fname = os.path.join(plotParams['PROJECTDIR'],folderStr,filestr + '%05d.bin' % (i))
        # Open file
        fid = open_file(fname, 'rb')
        temp = np.zeros((qu.ny, qu.nx, nvert))
        # Read header
        np.fromfile(fid, dtype=np.float32, count=1)
        if is_3d == 0:
            var = np.fromfile(fid, dtype=np.float32, count=qf.indexing.num_cells)[:]
            # http://scipy-cookbook.readthedocs.io/items/Indexing.html
            index = [qf.indexing.ijk[::, 1], qf.indexing.ijk[::, 0], qf.indexing.ijk[::, 2]]
            temp[index] = var
            #DRAWFIRE CHANGE
        else:
            temp = np.zeros((qu.ny, qu.nx, nvert))
            for k in range(0, nvert):
                t = np.fromfile(fid, dtype=np.float32, count=(qu.nx) * (qu.ny))[:]
                temp[::1, ::1, k] = np.reshape(t, (qu.ny, qu.nx))
        outvar.append(temp)
        fid.close()
    outvar = np.moveaxis(np.asarray(outvar),3,1)

    return outvar

def read_fire_grid(fid, qu, qf,projPath):
    fid.readline()  # ! FIRE GRID
    qf.nz = get_line(fid, 1)
    ratiox = get_line(fid, 1)
    ratioy = get_line(fid, 1)
    qf.nx = qu.nx * ratiox
    qf.ny = qu.ny * ratioy
    qf.dx = qu.dx / float(ratiox)
    qf.dy = qu.dy / float(ratioy)
    dz_flag = get_line(fid, 1)
    if dz_flag == 0:
        dztemp = get_line(fid, 2)
        qf.dz = dztemp * np.ones((qf.nz,), dtype=np.float32)
    else:
        qf.dz = np.zeros((qf.nz,), dtype=np.float32)
        for i in range(0, qf.nz):
            qf.dz[i] = get_line(fid, 2)

    qf.z = np.zeros((qf.nz + 1,))
    for k in range(1, qf.nz + 1):
        qf.z[k] = qf.z[k - 1] + qf.dz[k - 1]

    qf.zm = np.zeros((qf.nz,))
    for k in range(0, qf.nz):
        qf.zm[k] = qf.z[k] + qf.dz[k] * 0.5

    qf.horizontal_extent = [0., qf.dx * float(qf.nx), 0., qf.dy * float(qf.ny)]

    read_fire_grid_indexing(qf,projPath)

def read_fuel(fid):
    fid.readline()  # ! firetec fuel type
    fid.readline()  # ! stream type
    fid.readline()  # ! FUEL
    # - fuel density flag
    dens_flag = get_line(fid, 1)
    if dens_flag == 1:
        fid.readline()  # read density

    # - moisture flag
    if get_line(fid, 1) == 1:
        fid.readline()


    if dens_flag ==1:
        # - height flag
        fid.readline()
        fid.readline()
    
def set_line_fire(fid, ignitions, qf):
    x0 = get_line(fid, 2)
    y0 = get_line(fid, 2)
    len_x = get_line(fid, 2)
    len_y = get_line(fid, 2)

    iis = math.ceil(x0 / qf.dx)
    if x0 % qf.dx == 0:
        iis += 1
    iie = math.ceil((x0 + len_x) / qf.dx)
    jjs = math.ceil(y0 / qf.dy)
    if y0 % qf.dy == 0:
        jjs += 1
    jje = math.ceil((y0 + len_y) / qf.dy)

    iis = int(iis - 1)
    iie = int(iie - 1)
    jjs = int(jjs - 1)
    jje = int(jje - 1)

    ignitions.hor_plane[jjs:jje:1, iis:iie:1] = 1


def set_square_circle(fid, ignitions, qf):
    x0 = get_line(fid, 2)
    y0 = get_line(fid, 2)
    len_x = get_line(fid, 2)
    len_y = get_line(fid, 2)
    width_x = get_line(fid, 2)
    width_y = get_line(fid, 2)

    idelta = math.ceil(width_x / qf.dx)
    jdelta = math.ceil(width_y / qf.dy)
    iis = math.ceil(x0 / qf.dx)
    if x0 % qf.dx == 0:
        iis += 1
    iie = math.ceil((x0 + len_x) / qf.dx)
    jjs = math.ceil(y0 / qf.dy)
    if y0 % qf.dy == 0:
        jjs += 1
    jje = math.ceil((y0 + len_y) / qf.dy)

    iis = int(iis - 1)
    iie = int(iie - 1)
    jjs = int(jjs - 1)
    jje = int(jje - 1)
    idelta = int(idelta)
    jdelta = int(jdelta)

    for i in range(iis, iie):
        # bottom
        for j in range(jjs, jjs + jdelta - 1):
            ignitions.hor_plane[j, i] = 1
        # top
        for j in range(jje - jdelta + 1, jje):
            ignitions.hor_plane[j, i] = 1
    for j in range(jjs, jje):
        # right
        for i in range(iis, iis + idelta - 1):
            ignitions.hor_plane[j, i] = 1
        # left
        for i in range(iie - idelta + 1, iie):
            ignitions.hor_plane[j, i] = 1


def set_circle(fid, ignitions, qf):
    x0 = get_line(fid, 2)
    y0 = get_line(fid, 2)
    len_x = get_line(fid, 2)
    len_y = len_x
    width_x = get_line(fid, 2)

    iis = math.ceil(x0 / qf.dx)
    if x0 % qf.dx == 0:
        iis += 1
    iie = math.ceil((x0 + len_x) / qf.dx)
    jjs = math.ceil(y0 / qf.dy)
    if y0 % qf.dy == 0:
        jjs += 1
    jje = math.ceil((y0 + len_y) / qf.dy)

    iis = int(iis - 1)
    iie = int(iie - 1)
    jjs = int(jjs - 1)
    jje = int(jje - 1)

    radius = len_x * 0.5
    xringcenter = x0 + radius
    yringcenter = y0 + radius
    for j in range(jjs, jje):
        y = (float(j) - 0.5) * qf.dy
        for i in range(iis, iie):
            x = (float(i) - 0.5) * qf.dx
            dist = math.sqrt(math.pow(x - xringcenter, 2) + math.pow(y - yringcenter, 2))
            if radius - width_x <= dist <= radius:
                ignitions.hor_plane[j, i] = 1

def set_firetech_ignitions(ignitions, qf):
    sel_ign = np.zeros((qf.ny, qf.nx, qf.nz))
    fname = 'ignite_selected.dat'
    nelem = int(os.path.getsize(fname) / (5 * 4))
    fid = open_file(fname, 'rb')
    var = np.fromfile(fid, dtype=np.int32, count=nelem * 5)
    var = np.reshape(var, (5, nelem), order='F')
    var -= 1
    myindex = [var[2], var[1], var[3]]
    sel_ign[myindex] = 1
    fid.close()
    ignitions.hor_plane = np.sum(sel_ign, axis=2)

def open_file(filename, howto):
    try:
        fid = open(filename, howto)
        return fid
    except IOError:
        print("Error while opening " + filename)
        input("PRESS ENTER TO CONTINUE.")
        sys.exit()

def get_line(fid, datatype):
    return split_string(fid.readline(), datatype)

def split_string(s, datatype):
    # http://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
    s = s.strip()
    out = []
    for t in s.split():
        try:
            if datatype == 1:
                out.append(int(t))
            else:
                out.append(float(t))
        except ValueError:
            pass    
    return out[0]

def read_vertical_grid(qu, projPath):
    global folderStr
    fid = open_file(os.path.join(projPath,folderStr,'z_qu.bin'), 'rb')

    # Header
    np.fromfile(fid, dtype=np.int32, count=1)
    # Read z
    qu.z = np.fromfile(fid, dtype=np.float32, count=qu.nz + 2)

    # Header
    np.fromfile(fid, dtype=np.int32, count=2)

    # Read zm
    qu.zm = np.fromfile(fid, dtype=np.float32, count=qu.nz + 2)

    fid.close()

####################################
####    PYVISTA FUNCTIONALITY   ####
####################################

def buildGrid(data, heightMap,qu,qf,nGhost,IS3D):

    dims = data.shape   #cell-centered
    if(len(dims)== 2):
        dim = 2
    else:
        dim = dims[0]+1

    x = np.linspace(0.0, qu.nx * qu.dx, qu.nx+1)
    y = np.linspace(0.0, qu.ny * qu.dy, qu.ny+1)
    x,y = np.meshgrid(x,y)
    if(IS3D):
        Xs = np.repeat(x[np.newaxis, :, :], dim, axis=0)
        Ys = np.repeat(y[np.newaxis, :, :], dim, axis=0)
        grid = pv.StructuredGrid(Xs,Ys,heightMap)
    else:
        grid = pv.StructuredGrid(x,y,heightMap[0,:,:])

    return grid


def read_input(file_name, dim_xyz):

    nx, ny, nz = dim_xyz
    size = nx*ny*nz
    fid = open_file(file_name, 'rb')
    #read header & ignore it
    np.fromfile(fid, dtype=np.int32, count=1)
    data = np.fromfile(fid, dtype=np.float32, count=size)[:]
    # plt.plot(data)
    # plt.show()
    if(nz ==1):
        data = np.reshape(data,(ny,nx))
    else:
        data = data.reshape((nz, ny, nx))

    # return 3D view of flattened input
    return data

def ReadSensorData(dataFile,qu,zF):
    fid = open_file(dataFile,'r')
    windList = []
    for line in fid.readlines():
        windList.append(line.split())
    fid.close()
    iRead=0
    sensorNames, sensorData = ReadSensorWinds()
    nWinds = len(sensorNames)
    xs = np.zeros(nWinds)
    ys = np.zeros(nWinds)
    zs = np.zeros(nWinds)
    us = np.zeros(nWinds)
    vs = np.zeros(nWinds)
    while(iRead < nWinds):
        for i in range(nWinds+1):
            #check that name is in name list
            if(windList[i][0]==sensorNames[iRead]):
                xs[iRead]=np.real(windList[i][1])
                ys[iRead]=np.real(windList[i][2])
                xs[iRead]=xs[iRead]-3333.0
                ys[iRead]=ys[iRead]-2222.0
                #interpolate z value
                zi = np.int(np.floor(xs[iRead]/qu.dx))
                zj = np.int(np.floor(ys[iRead]/qu.dy))
                zir = (xs[iRead]%qu.dx)/qu.dx
                zjr = (ys[iRead]%qu.dy)/qu.dy
                zs[iRead]=zF[zj,zi]+ (zF[zj,zi+1]-zF[zj,zi])*zir+(zF[zj+1,zi]-zF[zj,zi])*zjr
                iRead += 1

    points = np.vstack((xs,ys,zs)).T
    winds = np.vstack((us,vs,np.zeros(len(us)))).T
    wdGrid = pv.PolyData(points)
    #wdGrid['winds']=winds
    return sensorNames, sensorData, wdGrid

def ReadSensorWinds():
    import pandas

    #builds dictionary with named sensors and vector stacks of wind per time
    windDir = '/Users/Drobinson/CSES/Rotated True N/'
    #day = 'Seventh/Post-move'
    day = 'Sixth'
    nameList = []
    initialize = 1
    for i in range(1,18):
        nameList.append('%02.d'%i)
    nameList.append('L1')
    nameList.append('L2')
    #12 missing from set
    nameList.remove('12')
    nameList.remove('04')
    timeData = {}

    for sensorName in nameList:
        filename = windDir+day+'/'+'Station'+sensorName+'_2019-11-01Average.csv'
        sensData = pandas.read_csv(filename)
        nWinds = len(sensData['U[m/s]'])
        us = sensData['U[m/s]']
        vs = sensData['V[m/s]']
        ws = np.zeros(len(vs))
        timeData[sensorName]=np.vstack((us,vs,ws)).T

    return nameList, timeData


def GenerateMovieFire(movieName,qu,qf,flags,zFuel,zF,terrain,topoDel,topo_cmap,plotParams,*args,**kwargs):
    #Plotter settings
    stride = 40
    pStride = 2
    PLOTMODE = kwargs.get('PLOTMODE', 0)
    PLOTMODE = plotParams['PLOTTING']
    try:
        interaction = np.int(plotParams['INTERACTIVE'])
    except:
        interaction = 1
        print('No INTERACTION argument supplied in input file. Interactive default to value of 1')

    pv.set_plot_theme('default')
    try:
        PLACECAM = np.int(plotParams['PLACECAM'])
    except:
        PLACECAM = 0
    try:
        PROJECTXY = np.int(plotParams['XYVIEW'])
    except:
        PROJECTXY = 0
    try:
        frameRate = np.int(plotParams['FRAMERATE'])
    except:
        frameRate = 1
    if(PLOTMODE=='SUB'):
        if(interaction or PLACECAM):
            plotter=pv.Plotter(shape=(2,2),off_screen=False,window_size=[1280,1024])
        else:
            plotter=pv.Plotter(shape=(2,2),off_screen=True,window_size=[1280,1024])
        plotter.add_axes()
    else:
        if(interaction or PLACECAM):
            plotter=pv.Plotter(window_size=[1280,1024])
        else:
            plotter=pv.Plotter(off_screen=True,window_size=[1280,1024])
    plotter.add_axes()
    if(PLOTMODE=='IND'):
        plotter.open_movie(plotParams['OUTDIR']+movieName,framerate=frameRate)
    else:
        if(os.path.exists(plotParams['OUTDIR'])):
            plotter.open_movie(plotParams['OUTDIR']+plotParams['TITLE'],framerate=frameRate)
        else:
            os.mkdir(plotParams['OUTDIR'])
            plotter.open_movie(plotParams['OUTDIR']+plotParams['TITLE'],framerate=frameRate)


    #Color Mapping stuff
    cbarArgs = dict(
        #interactive=True,
        title_font_size=35,
        label_font_size=29,
        #label_font_color='white',
        shadow=True,
        n_labels=3,
        fmt="%.2f",
        italic=True,
        font_family='arial',
    )

    tFont = 30
    #Read Data
    if(flags.fuel_density):
        fuelData = read_fireca_field("fuels-dens-", qf.ntimes, qf.time, qf, 0,plotParams)
        fMax = np.max(fuelData[0])/2.0
        fMax = 1.5
        #fOpacity = [0,0.5,0.7,0.8,0.9]
        #flipped because colorbar is flipped
        fOpacity = [0.9,0.8,0.7,0.5,0.2]
        fBounds = [0,1.5*np.mean(fuelData[0][np.nonzero(fuelData[0])])]
        fuel_cmap = ['black','brown','yellow','yellow','greenyellow','greenyellow','forestgreen','darkgreen']
        #fuel_cmap = ['black','darkorange','yellow','yellow','greenyellow','greenyellow','forestgreen','darkgreen']
        fTemp0 = fuelData[0][:,:,:]
        fTemp0[fTemp0 == 0.0]=-1.0
    if(flags.en2atm):
        enerData = read_fireca_field("fire-energy_to_atmos-", qf.ntimes, qf.time, qf, 1,plotParams)
        eMax = 200
        eOpacity = [0,0.6,0.7,0.8,0.9]
    if(flags.moisture):
        moistData = read_fireca_field("fuels-moist-", qf.ntimes, qf.time, qf, 0,plotParams)
        mMax = np.max(moistData[0])
        mMax = 1.0
        mOpacity = [0.9,0.8,0.7,0.5,0.2]
    if(flags.qu_qwinds_inst):
        winduData = read_fireca_field("qu_windu", qf.ntimes, qf.time, qf, 1,plotParams,layers=qu.nz)
        windvData = read_fireca_field("qu_windv", qf.ntimes, qf.time, qf, 1,plotParams,layers=qu.nz)
        windwData = read_fireca_field("qu_windw", qf.ntimes, qf.time, qf, 1,plotParams,layers=qu.nz)
        windGrid = buildGrid(winduData[0],zF,qu,qf,0,IS3D=True)
        wind_cmap = plt.cm.get_cmap("gist_yarg")



        #Check if quivers should be plotted
        try:
            QUIVERS = np.int(plotParams['QUIVERS'])
        except:
            QUIVERS = 0

        if(QUIVERS):
            try:
                qDensity = np.float(plotParams['QUIVERDENSITY'])
            except:
                qDensity = 0.5
            pStride = np.int(1.0/qDensity)
            #Check how colorbar should be scaled
            #ONLY FOR SURFACE QUIVERS FOR NOW
            try:
                #Can be 'AUTO','NONE','CUSTOM','FREE'
                WINDBAR = plotParams['WINDBARNORMALIZED']
            except:
                WINDBAR = 'FREE'
            if(WINDBAR=='CUSTOM'):
                try:
                    #Can be 'AUTO','NONE','CUSTOM','FREE'
                    wBarString= plotParams['WINDBARBOUNDS']
                    wBarString = wBarString.split(",",1)
                    WINDBARBOUNDS = [np.float(wBarString[0].lstrip('[')),np.float(wBarString[1].rstrip(']'))]
                except:
                    print('Winds colorbar setting set to CUSTOM but incorrect or missing WINDBARBOUNDS argument')
                    WINDBARBOUNDS = [0.0,10.0]
                    print('Bounds defaulted to ',WINDBARBOUNDS)
            if(WINDBAR=='AUTO'):
                print('Calculating Winds Color Bar Limits')
                lowLimit = 0.0
                highLimit = 0.0
                for i in range(1,qf.ntimes):
                    windMags = np.sqrt(winduData[i][0,::pStride,::pStride]**2.0+ \
                        windvData[i][0,::pStride,::pStride]**2.0 + \
                        windwData[i][0,::pStride,::pStride]**2.0  )
                    lowLimit = np.minimum(lowLimit,min(windMags.flatten()))
                    highLimit = np.maximum(highLimit,max(windMags.flatten()))
                WINDBARBOUNDS = [lowLimit,highLimit]
                print('Automated Wind Bounds',WINDBARBOUNDS)
            if(WINDBAR=='NONE'):
                cBarVis = False
            else:
                cBarVis = True



            #Plot Ground Winds
            zT = zF[2,1:,1:]+10 #offset to get above topo
            x = np.linspace(qu.dx/2.0, (qf.nx-0.5) * qu.dx, zT[::pStride,::pStride].shape[1])
            y = np.linspace(qu.dy/2.0, (qf.ny-0.5) * qu.dy, zT[::pStride,::pStride].shape[0])
            X,Y = np.meshgrid(x,y)
            quiverGrid = pv.StructuredGrid(X,Y,zT[::pStride,::pStride])
            uvectors = np.vstack((
                winduData[0][0,::pStride,::pStride].flatten(order='F'),
                windvData[0][0,::pStride,::pStride].flatten(order='F'),
                windwData[0][0,::pStride,::pStride].flatten(order='F'),
            )).T
            quiverGrid.vectors = uvectors
            try:
                qFactor = np.float(plotParams['QUIVERSCALE'])
            except:
                qFactor = 1
            if(PLOTMODE=='SUB'):
                plotter.subplot(0,1)
                plotter.add_text('Surface Winds',font_size=tFont,color='k')

            #Can be 'AUTO','NONE','CUSTOM','FREE'
            if(WINDBAR == 'CUSTOM' or WINDBAR == 'AUTO'):
                plotter.add_mesh(quiverGrid.glyph(factor=qFactor),stitle='Surface Winds [m/s]',scalar_bar_args=cbarArgs,lighting=True,ambient=0.3,diffuse=0.5, \
                    clim=WINDBARBOUNDS)
            if(WINDBAR == 'FREE' or WINDBAR == 'NONE'):
                plotter.add_mesh(quiverGrid.glyph(factor=qFactor),stitle='Surface Winds [m/s]',scalar_bar_args=cbarArgs,lighting=True,ambient=0.3,diffuse=0.5, \
                    show_scalar_bar=cBarVis)


    #build fuels grid based on existence of any data, but only do it once
    if('fuelData' in locals()):
        fuelGrid = buildGrid(fuelData[0],zFuel,qu,qf,nGhost,IS3D=True)
        if(qf.nz > 1):
            nGround = range(0,fuelGrid.number_of_cells-qf.nz+1,qf.nz)
            gGrid = fuelGrid.extract_cells(nGround)
            nCanopy = range(0,fuelGrid.number_of_cells+1)
            nCanopy = np.delete(nCanopy,nGround)
            cGrid = fuelGrid.extract_cells(nCanopy)
        else:
            gGrid = fuelGrid
    elif('enerData' in locals()):
        fuelGrid = buildGrid(enerData[0],zFuel,qu,qf,nGhost,IS3D=True)
    elif('moistData' in locals()):
        fuelGrid = buildGrid(moistData[0],zFuel,qu,qf,nGhost,IS3D=True)

    if(PLOTMODE=='SUB'):
        plotter.subplot(1,1)
        plotter.add_text('Topography',font_size=tFont,color='k')
        #add topo skirt
        plotter.add_mesh(terrain)
        #add topo coloring
        hValues = topoDel.points[:,2]
        #hValues[hValues<0]=0.0
        topoDel.point_arrays['heights']=hValues
        topoDel.set_active_scalars('heights')
        contours = topoDel.contour(10)
        plotter.add_mesh(topoDel,scalars=hValues,clim=[0.0,np.max(hValues)],cmap=topo_cmap,flip_scalars=True,scalar_bar_args=cbarArgs,show_scalar_bar=True)
        plotter.add_scalar_bar(title='Elevation [m]',title_font_size=cbarArgs['title_font_size'], \
           label_font_size=cbarArgs['label_font_size'], \
           shadow=cbarArgs['shadow'],\
           n_labels=cbarArgs['n_labels'],\
           fmt=cbarArgs['fmt'],\
           italic=cbarArgs['italic'],\
           font_family=cbarArgs['font_family'])
        #plotter.add_mesh(topoDel,cmap=topo_cmap,clim=[0,np.max(zFloor)],scalar_bar_args=cbarArgs,specular=0.5,specular_power=150,diffuse=0.5,ambient=0.5)
        if(flags.qu_qwinds_inst):
            plotter.subplot(0,1)
            plotter.add_mesh(topoDel,cmap=topo_cmap,scalar_bar_args=cbarArgs,specular=0.5,specular_power=150,diffuse=0.5,ambient=0.5)
            plotter.show_grid(font_size=30,xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
        if(flags.fuel_density):
            plotter.subplot(0,0)
            plotter.add_text('Fuels',font_size=tFont,color='k')
            fTemp = fuelData[0][:,:,:]
            fTemp[fTemp == 0.0]=-1.0
            if(qf.nz > 1):
                cGrid.cell_arrays['Fuel Density'] = fTemp[1:,:,:].flatten(order='F')
                cGrid.set_active_scalars('Fuel Density')
                fThreshed = cGrid.threshold([0.1,10.0*fMax])
                if(fThreshed.n_points > 0):
                    #oldcmap was summer
                    plotter.add_mesh(fThreshed,opacity=1.0,cmap='Greens',clim=[0.0,5.0],show_scalar_bar=False,flip_scalars=True,scalar_bar_args=cbarArgs,ambient=0.4)
                #enable better shading for fuels
                plotter.enable_eye_dome_lighting()
            else:
                gGrid.cell_arrays['Fuel Density'] = fTemp[:,:,:].flatten(order='F')
                gGrid.set_active_scalars('Fuel Density')
                fThreshed = gGrid.threshold([0.01,10.0*1.0])
                if(fThreshed.number_of_points > 0):
                    #oldcmap was summer
                    plotter.add_mesh(fThreshed,clim=[-0.2,2.0],cmap=fuel_cmap,show_scalar_bar=False,scalar_bar_args=cbarArgs)
            #add grid
            plotter.show_grid(font_size=30,xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
        if(flags.moisture):
            plotter.subplot(1,0)
            plotter.add_text('Moisture Content',font_size=tFont,color='k')
            fuelGrid.cell_arrays['Moisture'] = moistData[0][:,:,:].flatten(order='F')
            fuelGrid.set_active_scalars('Moisture')
            mThreshed = fuelGrid.threshold([1E-6,2.0])
            if(mThreshed.n_points > 0):
                plotter.add_mesh(mThreshed,opacity=0.4,clim=[0,1.0],scalar_bar_args=cbarArgs)
            #add grid
            plotter.show_grid(font_size=30,xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
        plotter.link_views()
    else:
        #add topo skirt
        plotter.add_mesh(terrain)
        #add topo coloring
        hValues = topoDel.points[:,2]
        #hValues[hValues<0]=0.0
        topoDel.point_arrays['heights']=hValues
        topoDel.set_active_scalars('heights')
        contours = topoDel.contour(10)
        plotter.add_mesh(topoDel,scalars=hValues,clim=[0.0,np.max(hValues)],cmap=topo_cmap,flip_scalars=True,scalar_bar_args=cbarArgs,show_scalar_bar=True)
        plotter.add_mesh(contours,show_scalar_bar=False)
        #plotter.add_mesh(topoDel,cmap=topo_cmap,scalar_bar_args=cbarArgs,specular=0.5,specular_power=150,diffuse=0.5,ambient=0.5)
        #add grid
        plotter.show_grid(font_size=30,xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')



    ########WIND SLICES########
    if(flags.qu_qwinds_inst and plotParams['WINDSLICES']=='1'):
        try:
            sOpacity = np.float(plotParams['SLICEOPACITY'])
        except:
            sOpacity = 0.15
        try:
            nSlices = np.int(plotParams['NSLICES'])
        except:
            nSlices = 5
        try:
            sliceAxis = plotParams['SLICEAXIS'].lower()
        except:
            sliceAxis = 'y'
        try:
            sliceWind = plotParams['SLICEWIND'].upper()
        except:
            sliceWind = 'W'
        if(PLOTMODE=='SUB'):
            plotter.subplot(0,1)
        if(sliceWind == 'U'):
            windGrid.cell_arrays[sliceWind] = winduData[0].flatten(order='F')
        if(sliceWind == 'V'):
            windGrid.cell_arrays[sliceWind] = windvData[0].flatten(order='F')
        if(sliceWind == 'W'):
            windGrid.cell_arrays[sliceWind] = windwData[0].flatten(order='F')

        windGrid.set_active_scalars(sliceWind)
        sliceends = windGrid.slice_along_axis(n=nSlices,axis=sliceAxis)
        plotter.add_mesh(sliceends,opacity=sOpacity,scalar_bar_args=cbarArgs,show_scalar_bar=True,cmap='jet',name='windSlice')#,clim = [-30,0])
    ########################



    #Set camera
    #<pos,focus,normal>
    try:
        cPosStr = plotParams['CAMPOSITION']
        cPosStr = cPosStr.split(",",2)
        cPos = (np.float(cPosStr[0].lstrip('<')),np.float(cPosStr[1]),np.float(cPosStr[2].rstrip('>')))
    except:
        cPos = (-0.5*(qf.ny*qu.dy),-0.5*qf.nx*qu.dx,10.0*np.max(zFloor)+20.0)
    try:
        cFocStr = plotParams['CAMFOCUS']
        cFocStr = cFocStr.split(",",2)
        cFoc = (np.float(cFocStr[0].lstrip('<')),np.float(cFocStr[1]),np.float(cFocStr[2].rstrip('>')))
    except:
        cFoc = topoDel.center
    try:
        cNormStr = plotParams['CAMNORMAL']
        cNormStr = cNormStr.split(",",2)
        cNorm = (np.float(cNormStr[0].lstrip('<')),np.float(cNormStr[1]),np.float(cNormStr[2].rstrip('>')))
    except:
        cNorm = (0,0,1)
    
    cVector = [cPos,cFoc,cNorm]
    plotter.camera_position = cVector
    plotter.camera_set = True

    #NECESSARY
    plotter.set_background("royalblue", top="aliceblue")
    path = plotter.generate_orbital_path(factor=1.1,n_points=36, shift=450) 
    if(PROJECTXY):
        #Top down projection
        plotter.enable_parallel_projection()
        plotter.enable_image_style()
        plotter.view_xy()
    if(interaction):
        plotter.show(auto_close=False)
    else:
        plotter.show(auto_close=False)

    try:
        ORBIT = np.int(plotParams['CAMORBIT'])
    except:
        ORBIT = 0

    if(ORBIT):
        plotter.orbit_on_path(path,write_frames=True)

    plotter.write_frame()
    try:
        TRACKFUEL = np.int(plotParams['FIRETRACKING'])
    except:
        TRACKFUEL = 0
    if(TRACKFUEL):
        trackX = []
        trackY = []
        trackZ = []
        try:
            cOffStr = plotParams['TRACKINGOFFSET']
            cOffStr = cOffStr.split(",",2)
            cOffset = (np.float(cOffStr[0].lstrip('<')),np.float(cOffStr[1]),np.float(cOffStr[2].rstrip('>')))
        except:
            cOffset = (200.0,200.0,300.0)
        try:
            trackHistory = np.int(plotParams['TRACKHISTORY'])
        except:
            trackHistory = 10
            print('No Fire-Tracking history length given. Default used: '+str(trackHistory))

    for i in range(1,qf.ntimes):
        if(PLOTMODE=='SUB'):
            for j in range(2):
                for k in range(2):
                    plotter.subplot(j,k)
                    plotter.clear()
        else:
            plotter.clear()
        print('Building Frame:',i,'/',qf.ntimes-1)

        if(flags.fuel_density):
            #fuelGrid.cell_arrays['Fuel Density'] = abs(fuelData[i][:,:,:]-fuelData[0][:,:,:]).flatten(order='F')
            #nCanopy = [i for i in nCanopy if i not in nGround]

            fTemp = fuelData[i][:,:,:]
            fTemp[fTemp == 0.0]=-1.0
            if(PLOTMODE=='SUB'):
                plotter.subplot(0,0)
                plotter.add_text('Fuels',font_size=tFont,color='k')
            if(qf.nz>1):
                cGrid.cell_arrays['Upper Consumption'] = abs((fTemp[1:,:,:]-fTemp0[1:,:,:])/fTemp0[1:,:,:]).flatten(order='F')
                cGrid.set_active_scalars('Upper Consumption')
                fConThreshed = cGrid.threshold([0.1,1])
                cGrid.cell_arrays['Fuel Density'] = fTemp[1:,:,:].flatten(order='F')
                cGrid.set_active_scalars('Fuel Density')
                fThreshed = cGrid.threshold([0.1,10.0*fMax])
                if(fThreshed.n_points > 0):
                    #plotter.add_mesh(fConThreshed,opacity=0.1,point_size=6.0,cmap='gist_yarg',clim=[0,1],scalar_bar_args=cbarArgs)
                    plotter.add_mesh(fThreshed,opacity=1.0,cmap='Greens',clim=[0.0,5.0],show_scalar_bar=False,flip_scalars=True,scalar_bar_args=cbarArgs,ambient=0.4)
                #ground
                gGrid.cell_arrays['% Consumed'] = abs((fTemp[0,:,:]-fTemp0[0,:,:])/fTemp0[0,:,:]).flatten(order='F')
                gGrid.set_active_scalars('% Consumed')
                gConThreshed = gGrid.threshold([0.1,1])
                gGrid.cell_arrays['Ground fuels'] = fTemp[0,:,:].flatten(order='F')
            else:
                gGrid.cell_arrays['% Consumed'] = abs((fTemp[:,:,:]-fTemp0[:,:,:])/fTemp0[:,:,:]).flatten(order='F')
                gGrid.set_active_scalars('% Consumed')
                gConThreshed = gGrid.threshold([0.01,1])
                gGrid.cell_arrays['Ground fuels'] = fTemp[:,:,:].flatten(order='F')
            gGrid.set_active_scalars('Ground fuels')
            gThreshed = gGrid.threshold([1E-6,10.0*1.0])
            if(gThreshed.n_points > 0 ):
                plotter.add_mesh(gThreshed,clim=[-0.2,2.0],show_scalar_bar=False,cmap=fuel_cmap)
            if(gConThreshed.n_points > 0):
                plotter.add_mesh(gConThreshed,opacity=0.2,point_size=6.0,show_scalar_bar=False,cmap='gist_yarg')
            if(TRACKFUEL):
                #find center of mass for consumed fuels for camera
                fuelShape = fuelData[i][:,:,:].shape
                fuel0 =fuelData[0][:,:,:]
                fTotal = 0.0
                iEst = 0
                jEst = 0
                for ii in range(fuelShape[2]):
                    for jj in range(fuelShape[1]):
                        for kk in range(fuelShape[0]):
                            if(fuelData[i][kk,jj,ii]>0.0):
                                fuelDiff =abs(fuelData[i][kk,jj,ii]-fuelData[i-1][kk,jj,ii]) 
                                fTotal += fuelDiff
                                iEst += fuelDiff*(ii+1)
                                jEst += fuelDiff*(jj+1)
                iEst = np.int(iEst/fTotal) - 1
                jEst = np.int(jEst/fTotal) - 1
                if(len(trackX)<trackHistory):
                    trackX.append(iEst)
                    trackY.append(jEst)
                else:#remove first and append last
                    trackX.remove(trackX[0])
                    trackY.remove(trackY[0])
                    trackX.append(iEst)
                    trackY.append(jEst)
            if(PLOTMODE=='SUB'):
                if(qf.nz>1):
                    plotter.enable_eye_dome_lighting()
                plotter.add_mesh(terrain)
            

        if(flags.en2atm):
            fuelGrid.cell_arrays['Energy to Atmos.'] = enerData[i][:,:,:].flatten(order='F')
            fuelGrid.set_active_scalars('Energy to Atmos.')
            eThreshed = fuelGrid.threshold([50,1E6])
            if(PLOTMODE=='SUB'):
                plotter.subplot(1,1)
                plotter.add_text('Energy',font_size=tFont,color='k')
                plotter.add_mesh(terrain)
            if(eThreshed.n_points > 0):
                plotter.add_mesh(eThreshed,opacity=eOpacity,cmap="autumn",clim=[0,eMax],show_scalar_bar=False,scalar_bar_args=cbarArgs,ambient=0.2)
            #find max of energy for camera
            # enerShape = enerData[i][:,:,:].shape
            # eTotal = 0.0
            # iEst = 0
            # jEst = 0
            # for ii in range(enerShape[2]):
            #     for jj in range(enerShape[1]):
            #         for kk in range(enerShape[0]):
            #             if(enerData[i][kk,jj,ii]>0.0):
            #                 eTotal += enerData[i][kk,jj,ii]
            #                 iEst += enerData[i][kk,jj,ii]*(ii+1)
            #                 jEst += enerData[i][kk,jj,ii]*(jj+1)
            # iEst = np.int(iEst/eTotal) - 1
            # jEst = np.int(jEst/eTotal) - 1


        if(flags.moisture):
            mTemp = moistData[i][:,:,:]
            mTemp[mTemp == 0.0]=-1.0
            mTempo = moistData[0][:,:,:]
            mTempo[mTempo == 0.0]=-1.0
            fuelGrid.cell_arrays['Moisture Loss'] = (abs(mTemp[:,:,:]-mTempo[:,:,:])/mTempo[:,:,:]).flatten(order='F')
            fuelGrid.set_active_scalars('Moisture Loss')
            mThreshed = fuelGrid.threshold([1E-6,mMax])
            if(PLOTMODE=='SUB'):
                plotter.subplot(1,0)
                plotter.add_text('Moisture Loss',font_size=tFont,color='k')
                plotter.add_mesh(terrain)
            if(mThreshed.n_points > 0):
                plotter.add_mesh(mThreshed,opacity=0.4,clim=[0,mMax],scalar_bar_args=cbarArgs)



        if(flags.qu_qwinds_inst):
            windGrid.cell_arrays['||DeltaW||'] = abs(windwData[i]-windwData[0]).flatten(order='F')
            windGrid.set_active_scalars('||DeltaW||')
            try:
                plumeViz = np.int(plotParams['PLUMEVIZ'])
            except:
                plumeViz = 0
            if(PLOTMODE=='SUB'):
                plotter.subplot(0,1)
                plotter.add_text('Winds',font_size=tFont,color='k')
                plotter.add_mesh(terrain)
            if(plumeViz):
                wThreshed = windGrid.threshold([1,10])
                if(wThreshed.n_points > 0):
                    plotter.add_mesh(wThreshed,opacity=0.1,clim=[0,10.0],cmap=wind_cmap,show_scalar_bar=False,scalar_bar_args=cbarArgs)
            plotter.show_grid(font_size=30,xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
            
            
        
            if(QUIVERS):
                uvectors = np.vstack((
                    winduData[i][0,::pStride,::pStride].flatten(order='F'),
                    windvData[i][0,::pStride,::pStride].flatten(order='F'),
                    windwData[i][0,::pStride,::pStride].flatten(order='F'),
                    #DIFFERNCE FROM INITIAL
                    # (winduData[i][0,::pStride,::pStride]-winduData[0][0,::pStride,::pStride]).flatten(order='F'),
                    # (windvData[i][0,::pStride,::pStride]-windvData[0][0,::pStride,::pStride]).flatten(order='F'),
                    # (windwData[i][0,::pStride,::pStride]-windwData[0][0,::pStride,::pStride]).flatten(order='F')
                )).T
                quiverGrid.vectors = uvectors
                #Can be 'AUTO','NONE','CUSTOM','FREE'
                if(WINDBAR == 'CUSTOM' or WINDBAR == 'AUTO'):
                    plotter.add_mesh(quiverGrid.glyph(factor=qFactor),stitle='Surface Winds [m/s]',scalar_bar_args=cbarArgs,lighting=True,ambient=0.3,diffuse=0.5, \
                        clim=WINDBARBOUNDS)
                if(WINDBAR == 'FREE' or WINDBAR == 'NONE'):
                    plotter.add_mesh(quiverGrid.glyph(factor=qFactor),stitle='Surface Winds [m/s]',scalar_bar_args=cbarArgs,lighting=True,ambient=0.3,diffuse=0.5, \
                        show_scalar_bar=cBarVis)
                #plotter.add_mesh(quiverGrid.glyph(factor=qFactor),stitle='Winds[m/s]',scalar_bar_args=cbarArgs,lighting=True,ambient=0.3,diffuse=0.5)



        if(PLOTMODE=='SUB'):
            plotter.subplot(1,0)
            plotter.add_mesh(terrain)
            plotter.add_text('Time:%ds'%qf.time[i],position='lower_left',font_size=40,font='arial')
        else:
            plotter.add_text('Time:%ds'%(qf.time[i]),position='lower_left',font_size=40,font='arial')
            plotter.add_mesh(terrain)
        if(PROJECTXY):
            #Top down projection
            plotter.enable_parallel_projection()
            plotter.enable_image_style()
            plotter.view_xy()
            #pl

        # CURRENT WIND SLICES########
        if(flags.qu_qwinds_inst and plotParams['WINDSLICES']=='1'):

            if(PLOTMODE=='SUB'):
                plotter.subplot(0,1)
            if(sliceWind == 'U'):
                windGrid.cell_arrays[sliceWind] = winduData[i].flatten(order='F')
            if(sliceWind == 'V'):
                windGrid.cell_arrays[sliceWind] = windvData[i].flatten(order='F')
            if(sliceWind == 'W'):
                windGrid.cell_arrays[sliceWind] = windwData[i].flatten(order='F')
            windGrid.set_active_scalars(sliceWind)
            sliceends = windGrid.slice_along_axis(n=nSlices,axis=sliceAxis)
            plotter.add_mesh(sliceends,opacity=sOpacity,scalar_bar_args=cbarArgs,show_scalar_bar=True,cmap='jet',name='windSlice')#,clim = [-30,0])
        ######
        plotter.set_background("royalblue", top="aliceblue")

        if(interaction):
            plotter.show(auto_close=False)

        if(TRACKFUEL):
            iEst = np.int(np.mean(trackX))
            jEst = np.int(np.mean(trackY))
            focHeight = zF[0,jEst,iEst]
            
            adapFoc = (iEst*qu.dx,jEst*qu.dy,  focHeight)
            camOffset = (adapFoc[0]+cOffset[0],adapFoc[1]+cOffset[1],adapFoc[2]+cOffset[2])
            plotter.camera_position = [camOffset,adapFoc,(0,0,1)]

        plotter.write_frame()

    if(ORBIT):
        plotter.orbit_on_path(path,write_frames=True)
    plotter.close()

    

def parseConfigFile(cFile):
    pList = {}
    inputFile = open(cFile,'r')
    for line in inputFile.readlines():
        if('=' in line):
            name, value = line.split("=",1)
            if(name.upper() in ['OUTDIR','TITLE','PROJECTDIR']):
                #dont change case of names
                pList[name.upper()]=value.rstrip()
            else:
                pList[name.upper()]=value.rstrip().upper()
    return pList





####################    
####    MAIN    ####
####################    

global topoDel, plotParams,terrainBox, quicVecStack, uQUIC,vQUIC,wQUIC,uQUICog,vQUICog,wQUICog
global folderStr
nGhost = 0      #number of ghost cells for fuel grid 



####Import Config File####
OG_PATH = os.getcwd()
CONFIG_PATH = os.path.join(OG_PATH,'VistaAargs.inp')
plotParams = parseConfigFile(CONFIG_PATH)
os.chdir('..')
plotParams['PROJECTDIR'] = os.getcwd()
print(plotParams)
movieName = 'standin.mp4'
#pv.start_xvfb()

####INPUT ARGUMENTS####
frameRate = 5
#plot modes: 0-all in one 1-Subplots,singleframe  2-Single plots of all output
try:
    plottingMode = plotParams['PLOTTING']    
except:
    #Default to including all
    plottingMode = 'ALL'    

#Check if legacy output or not
try:
    LEGACY = np.int(plotParams['LEGACYVERSION'])
except:
    LEGACY = 0

if(LEGACY):
    folderStr = ''
else:
    folderStr = 'Output'


qu, qf, ignitions, flags, fb = import_inputs(plotParams)
####    START TOPO    ####
#TOPO Always read even w/o topo
heights = read_input(os.path.join(plotParams['PROJECTDIR'],folderStr,'h.bin'), [qu.nx+2, qu.ny+2, 1])
#fix height corners (ghost cells at 0)
heights[0,0]=(heights[1,0]+heights[1,1]+heights[0,1])/3.0
heights[-1,0]=(heights[-2,0]+heights[-2,1]+heights[-1,1])/3.0
heights[0,-1]=(heights[1,-1]+heights[1,-2]+heights[0,-2])/3.0
heights[-1,-1]=(heights[-2,-1]+heights[-2,-2]+heights[-1,-2])/3.0
zg = read_input(os.path.join(plotParams['PROJECTDIR'],folderStr,'zg00000.bin'), [qu.nx, qu.ny, qu.nz+1])
zFloor = (heights[:-1,:-1]+heights[1:,:-1]+heights[:-1,1:]+heights[1:,1:])/4.0
zF = np.zeros((zg.shape[0],zFloor.shape[0],zFloor.shape[1]))
zF[:,1:-1,1:-1] =(zg[:,:-1,:-1]+zg[:,1:,:-1]+zg[:,:-1,1:]+zg[:,1:,1:])/4.0
zF[:,0,:]=zF[:,1,:]
zF[:,-1,:]=zF[:,-2,:]
zF[:,:,0]=zF[:,:,1]
zF[:,:,-1]=zF[:,:,-2]
#Build topo surface (2D)
x = np.linspace(0.0, qu.nx * qu.dx, qu.nx+1)
y = np.linspace(0.0, qu.ny * qu.dy, qu.ny+1)
x,y = np.meshgrid(x,y)
Xs = np.repeat(x[np.newaxis, :, :], 2, axis=0)
Ys = np.repeat(y[np.newaxis, :, :], 2, axis=0)
topoBox = np.repeat(zFloor[np.newaxis, :, :], 2, axis=0)-0.2
topoBox[0]=np.min(topoBox[1])-20.0
terrainBox = pv.StructuredGrid(Xs,Ys,topoBox)
terrainTop = pv.StructuredGrid(x,y,zFloor)


zFloor = zFloor+ 0.02*np.max(zFloor)
if(qu.isfire):
    #Build Fuel Corner grid
    #Build Fuel topo z heights grid
    zFuel = np.zeros((qf.nz+1,zFloor.shape[0],zFloor.shape[1]))
    zFuel[0,:,:]=zFloor[:,:]
    dz = qf.z[1]-qf.z[0]
    for k in range(1,qf.nz+1):
        zFuel[k,:,:] = zFloor[:,:] + dz*(k-1)


points = np.c_[x.reshape(-1),y.reshape(-1),(zFloor).reshape(-1)]
topo_cmap = plt.cm.get_cmap("gist_earth")


####    END TOPO    ####

if(plottingMode=='SUB' or plottingMode=='ALL'):
    GenerateMovieFire(movieName,qu,qf,flags,zFuel,zF,terrainBox,
        terrainTop,topo_cmap,plotParams,PLOTMODE=plottingMode)


if(plottingMode=='IND'):
    nFlag = 0
    flagList=[]
    for attr, value in flags.__dict__.items():
        if(value == 1):
            tFlags = copy.copy(flags)
            for att2, val2 in tFlags.__dict__.items():
                if(att2 != attr):
                    tFlags.__setattr__(att2,0)
            GenerateMovieFire(attr+'.mp4',qu,qf,tFlags,zFuel,zF,terrainBox,
                terrainTop,topo_cmap,plotParams,PLOTMODE=0)





