import serial
import serial.tools.list_ports
import time
import io

user_input = ['a3ae95868f9d929e928ba5a590959c968f949ba3b09793928ca0a6a69cb0a4ae9e9fae9e9994a8aa8f9d90aaa6909baed5e3b6af96c6d4bdc5d10e7f5dc3adb8',
                'b7d0a08392b09bb1998bbebe96a0aba0929ba9b8d4a39b998fb5bfbeabd2bacfb0b1d0afa99fc3c893b095c9c198acd01b34dccfa20215e6f50c7e7f7ff0c9e0',
                'c6f1b18699c6acc6a48fd8d6a0b2bbab99a5b8cbf3b6a8a498c9d5d2b6efcae9bebeedbcbeaed8e49bc19be4d9a6c0eb5371f5e5b5394200122e7f7f7f0bdcfe',
                'd51bc8899ee5c4deb08ff9f2aacbcbb7a1adcadd1cd0b8afa1e1efe8bf10d907cdcb12c6dbc0ef09a8d79e06f7bad90b7f7f0ffcce7f76182f507f7f7f25f022',
                'dd3de590a207e8f2bf931909b4ead5bfadb5dfe540f1cbb7acf3fff2bf25da17d7ce2fc6fed6fb29bae9a3220ed7f31c7f7f1303ed7f7f1b30527f7f7f26013d',
                'e84d0499a32a19fbd29e3213ba0bd5c1bbbcffdf5b15dfb6baf805ebb52bcd15decc41bb24eef942cff8af3417fb07197f7f00f90d7f7f0b193a7f7f7f171d53',
                'e3411da69e4a4cf3e6b4380dbf27c7bdc6c524e36132edadceecfbd1af1ab3fce5c93ebb3e11ec44dffdbd32071f0ef97f7fd4df267f5df6ec0f7f7f7ff33958',
                'ce1b27b59a6372d9f6d524fcd136b4b5c8cf4afb503df0a0e6e5edaebdfe96dbe1bf2dcc4343e12ee2f7c223e03b00e07f7fa9d93c7f26f8cefdf97f7fde5348',
                'b3e51ebfa36b7fb204f4fbe3e731b0acc7e462102330e694f6f4e692d5db89c0d0b115d7356fd207d2e6ba0db144fcf77f309af0447fe7f6b505c17f7fd86d16',
                'adb604c1ac5f7d9b1204cbc5f21bb5a1cd03650de90be68afb01e691ebc089b1b4b30bcc207cc9e8c6d5bafea53217197b05a6013a55bedca4f5007f7fe56fde',
                'b89cebb7ac3c679c22fea5bce9f8ba99d41b50eeb7e2f191f9fce7a1f7bb8fa7a8c105c7115dcee4d1cdc3f5a5152d2642e9b2ff1fecbcc594df4f7f7ff151b9',
                'bda7dcaaa70c4d9929ee8fc9d0d1bd95d52325c4a7d4eeaef1e4e0b8f9c698a1aeccf5d40028c9e5d8c2c1eaa7f82a1700caacf2efc5cfba90e37a7f7fde29b9',
                'c4bacca9a6e138ac24e691cac3b9b78ecc18eeb4b2cbe5cce5c0d1c3f4d09f99b1cbdcd3e1f0b1dcd1b0b1dcb8df08f3c2ae9fdfbac8dfc19de8777f7fae06dc',
                'cdc4bda4b1e225c619df9ac1cdc0a58cc202b7b4bfb6d8ddd7abc0bde9d1a38dafbdbec1bcc2a1d0be9b9ccfc3c6d9ca979ea5cb98c5e9cfa7e04a7f7f99edfa',
                'd2c0af9ac3f910c909d395b5d5cc9f95c4e59fb6c49cc6dcc5b9b4a8d8c4a680aca8a4a49ea1afc6aa8d9bc5bcaaaea98e9ba7c199bbebd6a9c7047f7fa8dc07',
                'c8b2a8a1cf0af9b1f5c390abd5d4aaa4c6caa4c1c286b4ccb2c5b4a9c3b2a980a9a59d92979cbdbca292a4bea99899968e99a0ba9fafe4d1a0a6cd7f5b9ecc00',
                'b2a1a4b2d50ee8a5dfb497a5cad7b5b2c1b4a3c9c388a9b79ac8bbbcb0a0aa80aab2a2929da4c4b19b9ca7b8929a99988a9395b29ca1dac19da0dc7f329dc4e8',
                '9a929ebed302e6c0cfa6989fbad3b8b7b79f97ccc98ea3a991c6c0c8b2a3a88ab1c0a793a3a8c6a8939fa6b68a9ca0a18b8d8aa9a1a3ccaea1b6007610aac8c6',
                '92909fbdc3ede3d4c8978f93a9cab0b0ad8f91ced198a1a2a2c6bbcac6b5a695bccdac8ea5a5c3ab9999a1b6989ea7a9948d89a4adadc19da1d11762f99fc6ad',
                '9e9aacb2acd4d7d7c69488879ebe9fa1a6889fd3dcafa59cafc6acc5dccaa798cad7b189a29bbfb7aa929eb9a59facb2a09392a6b4b0b89aa6e41f51f196bba7',
                'adadbaa894c0bdccc39f8b899cb59598a48aacdde8c9aa94b4c797c0ecdfa996d6e0bd979d8ebac5bb9da7bcac9cadbaad9c9baab6abb0abbae8173e01b2b2a9',
                'b9c7c6ae9bb0a1bac1a7918e9eb0a1a1a79cbbe8f5dfb28fb7c8a2c9f9efb9a8e0eaccab9d8fbbd1ccb4bac0ac8facc0b7a5a0aab39daabbcce0ff2b12d0b0a1',
                'c5e3d2c0aea98cadbdac968d9fb0aeaeb1b3cdf0feedc2a4c2d2c7e004ffd8cdecf3dcbda9a4c3dcdfd1cec6ad8cacc5bfa99ea6ae8ea4c6d1cbdd181de3ad93',
                'cffbe0d2bbae98afb9ae95889fb3b8bbc0cadcf300f6d8c5d4e4e9fa0f0ffbf1fcf9e4cabab9cfe4efeae0ccb4a2b7c7c1a5929fac96acc7cda7b6061bebab80',
                'd70ae8dcc3baaeb9b7ae9890a2b4bbc3ccd6e0eef9f9eae1eaf800081319120a04f8e3cfc6c8d6e7f7f8ebd1bfb7c4cac1a38b9cb1aebfc8c69cb3fe1af1b987',
                'db08e7d8c4c3bfc1b8aea1a1aab4b7c0ccd4d4dae8f0ededf60104030912140bfeebd6c5c2c6d1e0f4f8eacfbfc2ccccbfa89ca5bac3cdccc5b9d4011af8cf9c',
                'dcf6d9c7b9c1c4c3b6aba7abafafabb4c1c4bbbccddbdde1eef8f4e8eaf9fff7e7d3bdadaeb4bdcee4ede0c1b1bccbcbbfb1adb6c8d4d8cfcad0ee0e1f04e0b5',
                'd1dbc4aca0b3bfbfb0a3a2acaea698a1b0b09d98b2bfc0c3d5e0d5bebfd4ded6c8b7a390949da2b5d0dcd0ad92adc3c5baafb4c3d4ddd9cac3d4f816210cebc6',
                'bec1b1988aa6b5b7a69195a7a89e8893a4a4928ea3a99fa0bbc5bb9697b6beb8aca39788898b8ca4becdc6aa93a8bbbcafa3aec5d9dfd5b9a7c6f517210eecc2',
                'a1b2a7a09ba8b1b1a28d90a3a5a0959aa2a49fa0a19f8c90acb7b29e9caaaba1989594918d868ba1b8c5c6bbb3b6bbb5a18aa2c1d7dbcea990bdef0e1a07e6ab',
                'a3a1a6a9acb1b5b3aaa1a1a7aba9a7a7aaababa9a7a29ba0aeb8b9b4b0aca497898a9192919199a8b9c5cbcac6c4c0b5a497a8c3d5dad0bdb8d0f30c1305e5bf',
                'b08ea4a7b3b6bcb9b8b3b3b3b5b4b4b3b4b3b3b2b0afafb3bac0c3c0bab1a59789899095989da5b1bdc8ced1d0cdc7bfb7b7becdd9e0dddddeeefc0c0c09f2ed',
                'b585a2a4b4b7c0bfc3c1c2bfc0bdbdbabab8b8b7b9babdc2c7cbccc9c2b8aca09894959a9ea5acb6bec8cdd1d0cfcbcac8ccd0dae0e8eaf1f4fe030e0a0efe09',
                'be8ea3a4b1b5bfc1c7c6c8c5c4c0beb9b7b4b4b5b9bdc3c9ced0d1cdc7beb3aaa29e9d9fa1a7acb4bac1c5c9c8c9c8cacbd1d5dde1e7e9f0f2f9fa00fc02f906',
                'b99aa5a4adb0b8bcc2c3c5c2bfb9b5afaba8a7a9afb6bec5cbcfd0cdc8c0b7afa7a19e9d9ea2a6acb0b5b7b9b7b7b7bbbfc6cbd1d4d8d9dddee2e2e6e3e6e0e8',
                'aa9b9f9fa4a7adb2b7b8b9b6b2aca69f999391969ea8b2bbc2c6c8c6c2bcb4aba29b959293969a9ea2a6a6a6a3a09fa3a8b0b6babcbdbcbdbcbfbebfbebfbcbf',
                '96939393959a9fa5a9ababa8a49f99938d86848b949da7afb5babcbcb9b4ada49b938b84858a8f9396989896918a878e969da2a4a4a2a09f9e9fa0a1a09e9c9b',
                '88898685898e94999d9f9f9d9a96928f8d8c8e90959ba1a6abaeb0b0aeaba69f99928d8988898b8d9091918f8d89888c9094969694908c88888a8c8c8b8a8986',
                '84858384878c90949698989693908d8b8a8b8d9195989c9fa2a4a6a6a5a4a19d9995918e8c8b8b8c8d8f909090919292939392918e8b87848384868787858482',
                '8789898b8d8f9193949493918f8c888482868a8e919496999a9c9d9d9d9d9b9996938f8b888686888b8d8f9192939394939392908e8c8a888786858484848584',
                '8a8f8e8f90929293939392918e8c8a8888898b8d8f909192939495969697969593918e8a86838386898c8f919394949494949391908e8d8b8a89888887858583',
                '8c949294939493949393929291908f8f8e8e8e8e8d8d8d8d8d8d8e8f9091929291908e8d8b8b8a8b8d8e90919393949393929291908e8d8b8a89878584838481',
                '8c959394939493939393939392929191908f8e8d8c8a88878586888a8c8e8f9091919191919191919292939393939393939291908f8e8d8c8b89898787838480',
                '8c9291929191919191919091919190908f8f8d8c8a888684828385878a8c8e8f90919293939494949494939393929291908f8f8e8d8c8b8a8988878585848583',
                '898d8c8c8c8c8c8c8c8d8d8d8e8e8e8e8e8e8d8d8c8b8a8989898a8b8c8e8f9091929393949494949393929191908f8e8e8d8c8c8b8b8a898988888888868685',
                '8687878786868686878788898a8b8c8d8d8e8e8e8e8e8e8e8e8e8f8f90909191929292929292929191908f8e8d8c8b8a89888888878787868686858484848484',
                '83838382828181828384868788898a8c8d8d8e8f909090919191929292929292929291919190908f8e8d8c8b8a89888786858484848484848484858585838382',
                '82828282828283838485868788898a8c8d8e8e8f8f9090919191919191909090908f8f8e8e8d8d8c8b8a8a898887858483828181818181818282818080818180',
                '8385848485858686878889898a8b8c8c8d8e8e8f8f8f8f8f8f8f8f8f8e8e8e8d8d8c8c8c8b8b8b8a8a8989888887868685858483838383828382838483828181',
                '8587878888888889898a8a8b8b8c8c8d8d8d8e8e8e8e8d8d8d8c8c8b8b8a8a898989888888888888888888888787878686858585858484838383828181828382',
                '8889898989898a8a8a8b8b8c8c8c8c8d8d8d8d8d8d8c8c8b8b8a8989888787868685858585858686868787878787888888888787878787878786878888868686',
                '89888888898989898a8a8a8b8b8b8c8c8c8c8c8b8b8b8a8a8988878786858483828181818282838485858686878787878787888889888888888888878789898a',
                '88848584858687878888898a8a8a8b8b8b8b8b8b8b8a8a8a8989888887878685858484848484858686868787888888898a8a8a8a8a8b8b8c8c8c8d8f8e8d8d8e',
                '878082828484858587878888898a8a8a8b8b8b8b8b8b8b8b8b8a8a8a8a89898989898989898989898989898a8a8a8a8a8a8a8b8b8c8c8c8c8c8d8c8b8b8e8e90',
                '8b8788878888888989898a8a8b8b8b8c8c8c8c8d8d8d8d8d8d8d8d8d8d8d8d8d8d8d8d8d8c8c8d8d8c8c8c8d8d8c8d8d8e8e8d8d8d8e8e8e8e8f909191908f90',
                '8e8f8f8f8f8e8e8e8e8e8e8e8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8e8e8e8e8e8e8e8e8e8d8d8d8d8d8d8d8d8c8c8b8b8c8c8c8b8b8b8b8b8b898a8c8d8d',
                '919694949494949493939393939393939292929292919191919191909090908f8f8f8f8e8e8d8d8d8c8c8c8c8b8b8b8a8b8a8a898989898989898a8c8c898a8a',
                '91999799989897979797969696959595949494939292929291919090908f8e8e8e8d8d8c8c8b8b8a8a8989888887878685848483838281818181818081838684',
                '939b989998999898979797979696959594949393939292919190908f8f8f8e8e8d8d8d8c8b8b8b8a8a8989898887878788888786858686868686888a8b878988',
                '949998999898979797979595959594949393929291919090908f8e8e8e8d8d8c8c8c8b8b8b8b8a8a8a8a8a898a8a8a8989888a8a8b898a8a8a8a8987888b8d8c',
                '9599979796969696959495949393939291919090908f8e8e8d8d8d8c8c8c8b8b8a8a8a8a89898a8a8989898a8a898a8b8c8d8c8c8c8d8e8e8e8f919494909190',
                '939595969594949393939291919190908f8f8e8d8c8c8c8b8b8988888886868585848383838382828383848484868686858587888a89898a8a8a8985888d8f8e',
                '9396949393949393929192929090908f8e8e8e8e8e8c8c8b8a8b8b8a898a8988878787868585868685868687878788898c8d8b8b8a8d8e8e8f90949998919191',
                '929294959493939293939191929291919291908f8e909090908e8d8d8e8c8d8c8d8d8c8d8d8e8c8c8e8d8e8c8c8f8e8c89888c8d8f8b8a8c8c8b8680808d8f8e']

#max78000 feather board
PRODUCT_ID = 516 
DEVICE_ID = 3368
BAUD_RATE = 115200
CHUNKS_LEN = 64

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
            print("RX1: {}".format(line))
            print("TX: {}".format(chunk))
            
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
            current_state = IDLE_STATE
        
        elif current_state == IDLE_STATE:
            if False:
                print("Changing state to WRITING_CHUNK_STATE")
                hex_lines = user_input.copy() # set a new input
                current_state = WRITING_CHUNK_STATE
                start_time = time.time()
            

finally:
    ser.close()