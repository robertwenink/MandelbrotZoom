# Mandelbrot
# example:  https://godhalakshmi.medium.com/a-simple-introduction-to-the-world-of-fractals-using-python-c8cb859bfd6d
# animeren: https://matplotlib.org/matplotblog/posts/animated-fractals/
# TODO: https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set

####### IMPORT PACKAGES/LIBRARIES #########
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
from numba import cuda
from numba import njit, prange
import time

####### SETTINGS #########
X_RESOLUTIE = 1000
Y_RESOLUTIE = 1000
NR_FRAMES = 50
MAX_ITS = 1000
MODE = 2            # 0 for escape, 1 for histogram, 2 smooth

ANIMATE = True
LIVEPLOTTING = True
TRAJECTORY = [[-0.5,0,1]] # formulated as [x,y,zoom], 
SMOOTHING_POWER = 1.2 # om de locatieverplaatsing sneller te laten gaan dan de oppervlakte zoom, voor smoothness!, define minimaal 1.2

start_height = 3
start_width = 3 * (X_RESOLUTIE/Y_RESOLUTIE)

####### ADD TRAJECTORY POINTS #########
# zelf uitzoeken!
# http://www.jakebakermaths.org.uk/maths/mandelbrot/canvasmandelbrotv12bak7512.html

Z = [0.00164372197255,-0.822467633298876,128*4096]
TRAJECTORY.append(Z)
Z2 = [0.00164372197255,-0.822467633298876,3*4096**3]
TRAJECTORY.append(Z2)


def init_figure():
    # Create figure scladed for our given resolution
    fig, ax = plt.subplots(figsize=(X_RESOLUTIE/100, Y_RESOLUTIE/100),frameon=False) 

    # Stuff apparently needed to make the image borderless             
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

     # we willen geen grid of assenstelsel oid zien
    ax.axis("off")                                 

    # neppe data maken!        
    X = np.zeros((X_RESOLUTIE,Y_RESOLUTIE))    

    # lege image plotten, met colorscheme, en vanaf beneden beginnen met plotten (ipv vanaf boven)             
    img = ax.imshow(X.T, cmap='Spectral',origin='lower',vmin=0,vmax=255) 

    # als we liveplotting doen moeten we eerst de figure al tekenen!    
    if LIVEPLOTTING:
        plt.draw()                                              
        plt.pause(0.001)          

    return fig,ax,img

# figuur initialiseren
fig,ax,img = init_figure()

def get_trajectory_point(i):
    """
    Function to convert trajectory point from [x,y,zoom] -> [x,y,width,height]
    """
    [x,y,zoom] = TRAJECTORY[i]
    zoom_width = start_width / zoom
    zoom_height = start_height / zoom
    return x, y, zoom_width, zoom_height

@njit(parallel=True, fastmath=True)
def mandelbrot(x_cor, y_cor):
    """
    Escape time algorithm; simplest variant.
    """
    ESCAPE_RADIUS = 4 #
    X = np.zeros((X_RESOLUTIE,Y_RESOLUTIE),dtype="float64")

    for i in prange(X_RESOLUTIE):
        for j in prange(Y_RESOLUTIE):
            c = complex(x_cor[i],y_cor[j])
            z = complex(0, 0)
            n = 0.0
            for k in range(MAX_ITS):
                z = z*z + c
                n = n + 1
                if (abs(z) > ESCAPE_RADIUS):
                    break
            X[i,j] = n              
    return X

# https://developer.nvidia.com/blog/numba-python-cuda-acceleration/
@cuda.jit(device=True)
def mandel(x, y, max_iters):
  """
  Given the real and imaginary parts of a complex number,
  determine if it is a candidate for membership in the Mandelbrot
  set given a fixed number of iterations.
  """
  c = complex(x, y)
  z = 0.0j
  for i in range(max_iters):
    z = z*z + c
    if (z.real*z.real + z.imag*z.imag) >= 4:
      return i

  return max_iters

@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
  height = image.shape[0]
  width = image.shape[1]

  pixel_size_x = (max_x - min_x) / width
  pixel_size_y = (max_y - min_y) / height

  startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
  startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
  gridX = cuda.gridDim.x * cuda.blockDim.x;
  gridY = cuda.gridDim.y * cuda.blockDim.y;

  for x in range(startX, width, gridX):
    real = min_x + x * pixel_size_x
    for y in range(startY, height, gridY):
      imag = min_y + y * pixel_size_y 
      image[y, x] = mandel(real, imag, iters)

    # TODO dit in histogramcoloring opnemen.
    #  gimage = np.zeros((1024, 1536), dtype = np.uint8)
    X = np.zeros((X_RESOLUTIE,Y_RESOLUTIE),dtype="float64")
    blockdim = (32, 8)
    griddim = (32,16)

    d_image = cuda.to_device(gimage)
    mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 20) 
    d_image.to_host()

    imshow(gimage)

def histogramColoring(x_cor, y_cor):
    """
    Coloring schema that uses the original mandelbrot counts but relates them to all other counts, for smoother behaviour.
    """

    # pass 1, retrieve counts
    iteration_counts = mandelbrot(x_cor,y_cor)

    # pass 2, create bins filled with frequencies of the mandelbrot iteration counts; is of maximum size MAX_ITS
    (unique_its, inverese_indices, frequencies) = np.unique(iteration_counts, return_inverse = True, return_counts=True)

    # pass 3, count total, excluding the counts of pixels that reached bailout
    total = np.sum(frequencies[:-1])

    # pass 4, determine color/hue
    # basically, create a cumulative map of frequencies, divide by total, and index using location of a count in unique_its
    color_bins = np.cumsum(frequencies)/total
    color = np.take(color_bins,inverese_indices)

    return np.reshape(color,(X_RESOLUTIE,Y_RESOLUTIE))

    ## this is more like the implementation on wikipedia for step 4, laten staan voor amy
    # color = np.zeros_like(iteration_counts)
    # for index, count in np.ndenumerate(iteration_counts):
    #     for idx, i in enumerate(unique_its):
    #         if i > count:
    #             break
    #         color[index] += frequencies[idx]/total     

def animate(args):
    """
    Function that facilitates the animation.
    """
    (x, y) = (args[1], args[2])
    (width, height) = (args[3], args[4])

    x_cor = np.linspace(x-width/2,x+width/2,X_RESOLUTIE)
    y_cor = np.linspace(y-height/2,y+height/2,Y_RESOLUTIE)

    # op deze manier increase je de max_its per animatiestap!\
    # max_its = round(1.25**(args[0] + 5))  # calculate the current threshold

    if MODE == 0:
        X = mandelbrot(x_cor,y_cor)
    else:
        X = histogramColoring(x_cor,y_cor)

    # associate colors to the iterations with an iterpolation
    img.set_data(X.T)
    img.autoscale()
    if LIVEPLOTTING:
        fig.canvas.flush_events()

    return [img]

def next_trajectory_xy(trajectory_change_at_iteration,r_height):
    """
    Generator for returning the next points and factor for interpolation and smooth trajectory """
    i = 0

    for i in range(len(TRAJECTORY)-1):
        start_x, start_y, start_width, _  = get_trajectory_point(i)
        zoom_x, zoom_y, zoom_width, _  = get_trajectory_point(i+1)

        frame_interval = trajectory_change_at_iteration[i+1] - trajectory_change_at_iteration[i] 
        r_xy = r_height**SMOOTHING_POWER # ziet er iets soepeler uit als de locatie sneller convergeert dan de screen size

        # we zoeken naar (1+f)*r^N = f , ofwel fac begint met 1+f eindigt met -f, en corrigeren hiervoor bij interpoleren
        # herschrijven geeft f = r^N/(1-r^N)
        rN = r_xy**frame_interval 
        f = rN/(1-rN)
        facxy = 1 + f
        yield start_x,start_y,zoom_x,zoom_y,r_xy,facxy,f
        
def frames():
    """
    Generator for the animation, returning iteration number, new centers and screen size.
    """
    start_x, start_y, width, height  = get_trajectory_point(0)
    zoom_x, zoom_y, zoom_width_end, zoom_height_end = get_trajectory_point(-1)

    r_width = (zoom_width_end/start_width)**(1/(NR_FRAMES))
    r_height = (zoom_height_end/start_height)**(1/(NR_FRAMES))

    def intpol(start,end,fac,f):
        return start * (fac-f) + end * ((1+f)-fac)
    
    trajectory_change_at_iteration=[]
    fac_temp = 1 
    j = 0
    for i in range(NR_FRAMES):
        fac_temp*= r_height
        if fac_temp <= 1/(TRAJECTORY[j][-1]):
            trajectory_change_at_iteration.append(i)
            j += 1
        if j == (len(TRAJECTORY)-1):
            break   # voegt niet altijd de laatste toe, wss afrondfouten
    trajectory_change_at_iteration.append(NR_FRAMES-1)
            
    gen = next_trajectory_xy(trajectory_change_at_iteration,r_height)

    j = 0
    for i in range(NR_FRAMES):
        # Stel we verplaatsen x, y, width en height lineair via interpolation afhankelijk van factor fac.
        # height en width tezamen vormen het oppervlak.
        # om de oppervlak verandering linear te laten verlopen moet je dus fac**(1/2) doen
        # echter, stel je neem 10 stappen van 0.1 tussen 1.01 en 0.01, 
        # is de eerste stap een verschil van 10% en de laatste stap van 99%
        # Het verloop moet dus relatief uniform zijn ook om visueel uniform te zijn ofwel x^(n+1)=x^n * r met r constant.
        # we moeten dus een geometrische reeks hebben waarbij we voor k = NR_FRAMES een r vinden waarvoor geldt dat
        # start*r^NR_FRAMES = end ofwel r = kde wortel van end/start. De fac**1/2 is nu niet meer van belang.
        # Verder moet x en y hetzelfde verlopen om overeen te komen!
        
        if i == trajectory_change_at_iteration[j] and not i == trajectory_change_at_iteration[-1]:
            j += 1
            start_x,start_y,zoom_x,zoom_y,r_xy,facxy, f = next(gen)
            print("change of trajectory!")
        x = intpol(start_x,zoom_x,facxy,f)
        y = intpol(start_y,zoom_y,facxy,f)
        yield i, x, y, width, height

        facxy *= r_xy
        width *= r_width
        height *= r_height
        print("{:.2f}% complete".format((i+1)/NR_FRAMES*100))

def animateMandelbrot():
    """
    Function that calls the matplotlib animation function, and saves the result to a gif.
    """
    # TODO https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=120, blit=True, save_count = NR_FRAMES)
    anim.save('mandelbrot.gif',writer='pillow')
    print("{:.2f}% complete".format(100))

def showNormal():
    zoom_x, zoom_y, zoom_width, zoom_height = get_trajectory_point(-1)
    x_cor = np.linspace(zoom_x-zoom_width/2,zoom_x+zoom_width/2,X_RESOLUTIE)
    y_cor = np.linspace(zoom_y-zoom_height/2,zoom_y+zoom_height/2,Y_RESOLUTIE)
    if MODE == 0:
        X = mandelbrot(x_cor,y_cor)
    else:
        X = histogramColoring(x_cor,y_cor)
    

    img.set_data(X.T)
    img.autoscale()
    # dit behoudt niet het originele bestandsformaat, maar pakt m zoals weergegeven!!!
    # plt.savefig('mandelbrot.png', bbox_inches='tight', pad_inches = 0)
    data = ((X.T-X.min())/X.max()*255.0)
    plt.imsave('mandelbrot.png', data, cmap='Spectral',origin='lower',vmin=0,vmax=255)

start = time.perf_counter()
if ANIMATE:
    animateMandelbrot()
else:
    showNormal()
print("calc time: {}".format(time.perf_counter() - start))

plt.draw()
plt.pause(60)
# plt.show()
