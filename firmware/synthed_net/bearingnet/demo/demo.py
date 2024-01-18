import serial
import serial.tools.list_ports
import time
import io
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


#data for demo with hex images
zlistq = list(np.load("/Users/sam/Repositories/ethz/bearings-max78000/data/demo/zlistq_from_averaging_30_windows.npy").astype(np.int8))
input_imgs = list(np.load("/Users/sam/Repositories/ethz/bearings-max78000/data/demo/input_imgs_from_averaging_30_windows.npy").astype(np.uint8))

#data for demo with float windows
DECIMATE_INPUT_WINDOWS = 10 #so the demo is faster
EXPERIMENT = 1 #first bit of 1 is what we trained on (also we left out "early" in experiment 1, around 330 files), 2, 3 never seen before
BEARING = 3
input_windows = list(np.load("ims_bearings_all_exp%s_b%s_input_windows_from_averaging_30_windows.npy"%(EXPERIMENT, BEARING) ).astype(np.double)[::DECIMATE_INPUT_WINDOWS])

len_input_windows = len(input_windows)
len_input_imgs = len(input_imgs)


#for live plot
avg_windows_per_file = 4
x_size = (len_input_windows*DECIMATE_INPUT_WINDOWS)

x_vec = np.linspace(0,x_size//avg_windows_per_file,x_size//DECIMATE_INPUT_WINDOWS)[0:-1]


y_vec = np.zeros(len(x_vec))
line1 = []

print("Demo will run with {} input windows (demo 1 selected) or {} input images (if demo 2 selected)".format(len_input_windows, len_input_imgs))


#max78000 feather board
PRODUCT_ID = 516 
DEVICE_ID = 3368
BAUD_RATE = 115200

CHUNKS_LEN = 50
WINDOW_LEN = 200

def live_plotter(x_vec,y1_data,line1,processing_time=0,pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(30,12))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Health Indicator')
        plt.xlabel('File Index')
        plt.show()
        #plt.ylim([20+60,128+60])
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    plt.title('Dataset: NASA IMS, Bearing: {}, Experiment: {}, Time/Sample: {}ms'.format(BEARING, EXPERIMENT,int(processing_time)//1000))
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1


def convert_window_to_float_chunks(input_window):
    float_chunks = [input_window[:WINDOW_LEN//2], input_window[WINDOW_LEN//2:]]
    float_strings = []
    for float_chunk in float_chunks:
        float_string = ' '.join(['%f'% i for i in float_chunk])
        float_strings.append(float_string)
    
    return float_strings


def find_serial_port():
    """Finds and returns the first available serial port."""
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if port.pid == PRODUCT_ID and port.vid == DEVICE_ID:
            print("Found serial port: {}".format(port.device))
            return port.device
    return None

port = find_serial_port()
if port is None:
    raise Exception("No serial port found.")

# Establish serial connection
ser = serial.Serial(port, BAUD_RATE, timeout=1)
chunks_written = 0
float_chunks_written = 0
chunks_wrong = 0
j = 0

PROCESS_HEX_IMAGE_DEMO = 0
PROCESS_WIN_DEMO = 1

IDLE_STATE = 0
# DEMO 2 states (hex chunks of image)
READING_CHUNKS_STATE = 1
WRITING_CHUNK_STATE = 2
VERIFY_CHUNK_STATE = 3
PROCESS_OUTPUT_STATE = 4

# DEMO 1 states (long chunks of window)
WRITING_WIN_CHUNK_STATE = 11
VERIFY_WIN_CHUNK_STATE = 22
PROCESS_WIN_OUTPUT_STATE = 33


def reset(ser):
    ser.close()
    ser.open()
    time.sleep(3)
    
try:
    reset(ser)
    current_state = 0
    start_time = 0
    hex_lines = []
    float_lines = [] #i.e. line of float values 
    chunk = ""
    local_label = ""
    demo = PROCESS_WIN_DEMO
    
    ser.flush() # it is buffering. required to get the data out *now*
    hello = ser.readline()
    print(hello)

    ser.write(b"d\r")
    ser.flush()
    
    while True:
        ser.flush()
        line = ser.readline()
        line = line.decode("utf-8").strip()
        
        if "*** The sample input is: ****" in line and current_state == IDLE_STATE:
            current_state = READING_CHUNKS_STATE
            print("Reading sample input from device.")
            
        elif current_state == READING_CHUNKS_STATE:
            hex_lines.append(line)
            if len(hex_lines) >= CHUNKS_LEN:
                print("Read {} lines".format(len(hex_lines)))
                print("Changing state to WRITING_CHUNK_STATE")
                
                if demo == PROCESS_HEX_IMAGE_DEMO:
                    current_state = WRITING_CHUNK_STATE
                else:
                    print("Demo: PROCESS WINDOW")
                    time.sleep(3)
                    current_state = IDLE_STATE
                start_time = time.time()
                
        
        ## -- DEMO 1 -- ##
        elif current_state == WRITING_WIN_CHUNK_STATE and "float_chunk(" in line:
            float_chunk = float_lines.pop(0).strip()

            #send a chunk
            b = ("%s\r"%float_chunk)
            ser.write(b.encode("utf-8"))
            float_chunks_written += 1
            ser.flush()
            
            current_state = VERIFY_WIN_CHUNK_STATE
        
        elif current_state == VERIFY_WIN_CHUNK_STATE and "float_chunk(" not in line:
            if float_chunk != line:
                print("Chunk wrong!")
            else:
                #print("Chunk correct")
                pass
            
            if float_chunks_written >= 2:
                end_time = time.time()
                #print("Writing {} float chunks took {} seconds".format(float_chunks_written, end_time - start_time))
                float_chunks_written = 0
                current_state = PROCESS_WIN_OUTPUT_STATE
            else:
                current_state = WRITING_WIN_CHUNK_STATE
                
                
        elif len(float_lines) == 0 and "Output: " in line and current_state == PROCESS_WIN_OUTPUT_STATE:
            #print("Changing state to IDLE_STATE")
            output = line.split(' ')
            print(output)
            z1 = output[1]
            z2 = output[2]
            health_indicator = output[3]
            total_processing_time = output[4]
            end_time = time.time()
            
            #add to live plot
            y_vec[-1] = health_indicator
            line1 = live_plotter(x_vec,y_vec,line1, processing_time=total_processing_time)
            y_vec = np.append(y_vec[1:],0.0)
            
            current_state = IDLE_STATE
            
            print("Health indicator: {}".format(health_indicator))
            print("total_processing_time on uc in us: {}".format(total_processing_time))
            print("Total time: {} seconds".format(end_time - start_time))

                
        ## -- DEMO 2 -- ##
        
        elif len(hex_lines) == 0 and current_state == WRITING_CHUNK_STATE:
            #print("Changing state to IDLE_STATE")
            current_state = IDLE_STATE
        
        elif current_state == WRITING_CHUNK_STATE and "chunk(" in line:
            chunk = hex_lines.pop(0).strip()
            #print("RX1: {}".format(line))
            #print("TX: {}".format(chunk))
            
            #send a chunk
            b = ("%s\r"%chunk)
            ser.write(b.encode("utf-8"))
            chunks_written += 1
            ser.flush()
            
            current_state = VERIFY_CHUNK_STATE
            
        elif current_state == VERIFY_CHUNK_STATE and "chunk(" not in line:
            previous_chunk = chunk
            if(line!=chunk): chunks_wrong += 1
            
            if chunks_written >= CHUNKS_LEN:
                end_time = time.time()
                print("Writing {} chunks took {} seconds".format(chunks_written, end_time - start_time))
                print("All chunks written with {} wrong chunks".format(chunks_wrong))
                print("Changing state to IDLE_STATE")
                chunks_written = 0
                current_state = PROCESS_OUTPUT_STATE
            else:
                current_state = WRITING_CHUNK_STATE
        
        elif current_state == PROCESS_OUTPUT_STATE and "Output: " in line:
            print("From device: {}".format(line))
            print("Local label: {}".format(local_label))
            current_state = IDLE_STATE
        
        elif current_state == IDLE_STATE:
            if j < len_input_windows:
                
                if demo == PROCESS_HEX_IMAGE_DEMO:
                    image = input_imgs.pop(0)
                    local_label = zlistq.pop(0)
                    #in hex
                    local_label = ''.join(format(i, '02x') for i in local_label)
                    
                    hex_encoded_rows = []
                    for row in image[0]:
                        # Ensure that the row is flattened to a 1D array
                        flat_row = row.flatten()
                        # Convert each element in the flat_row to a 2-character hex string
                        hex_row = ''.join(format(i, '02x') for i in flat_row)
                        hex_encoded_rows.append(hex_row)
                    
                    hex_lines = hex_encoded_rows.copy() # set a new input, copy!
                    current_state = WRITING_CHUNK_STATE
                    print("Changing state to WRITING_CHUNK_STATE (%i of %i)" % (j, len_input_imgs) )
                    
                    
                else:
                    input_window = input_windows.pop(0)
                    float_encoded_chunks = convert_window_to_float_chunks(input_window)
                    float_lines = float_encoded_chunks.copy()
                    current_state = WRITING_WIN_CHUNK_STATE
                    print("Changing state to WRITING_WIN_CHUNK_STATE (%i of %i)" % (j, len_input_windows))
                    
                
                start_time = time.time()
                j += 1
            

finally:
    ser.close()