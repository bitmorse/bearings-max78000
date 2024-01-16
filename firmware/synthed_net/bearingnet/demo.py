import serial
import serial.tools.list_ports
import time
import io
import numpy as np

zlistq = list(np.load("/Users/sam/Repositories/ethz/bearings-max78000/data/demo/zlistq_from_averaging_30_windows.npy").astype(np.uint8))
input_imgs = list(np.load("/Users/sam/Repositories/ethz/bearings-max78000/data/demo/input_imgs_from_averaging_30_windows.npy").astype(np.uint8))

#max78000 feather board
PRODUCT_ID = 516 
DEVICE_ID = 3368
BAUD_RATE = 115200
CHUNKS_LEN = 50


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
chunks_wrong = 0
j = 0

IDLE_STATE = 0
READING_CHUNKS_STATE = 1
WRITING_CHUNK_STATE = 2
VERIFY_CHUNK_STATE = 3
PROCESS_OUTPUT_STATE = 4
    
def reset(ser):
    ser.close()
    ser.open()
    time.sleep(3)
    
try:
    reset(ser)
    current_state = 0
    start_time = 0
    hex_lines = []
    chunk = ""
    local_label = ""
    
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
                current_state = WRITING_CHUNK_STATE
                start_time = time.time()
        
        elif len(hex_lines) == 0 and current_state == WRITING_CHUNK_STATE:
            print("Changing state to IDLE_STATE")
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
            if j < 100:
                print("Changing state to WRITING_CHUNK_STATE")
                            
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
                start_time = time.time()
                j += 1
            

finally:
    ser.close()