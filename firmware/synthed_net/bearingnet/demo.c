#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "demo.h"
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

#include "wavelib.h"

#define DEMO_INPUT_LEN 50*50
#define DEMO_INPUT_CHUNK_LEN 50

#define DECIMATION 2

#define INTERP_SIZE 50
#define SAMPLE_LEN 200

#define SCALES 10
#define MORLET 4
#define DT 1
#define DJ 0.1*3
#define S0 2.5*DT
#define POW 2 
#define REFERENCE 0.09

#define PI 3.14159265358979323846

volatile uint32_t cnn_time; // Stopwatch

// Expected output of layer 4 for bearingnet given the sample input (known-answer test)
static const uint32_t sample_output[] = SAMPLE_OUTPUT; 


// 1-channel 50x50 data input (2500 bytes / 625 32-bit words):
// HWC 50x50, channels 0 to 0
static const uint32_t input_0[] = SAMPLE_INPUT_0;


static int32_t  cnn_output[(CNN_NUM_OUTPUTS + 3) / 4]; // CNN output data

uint32_t input_serial[DEMO_INPUT_LEN]; //the input from the serial port by the user

// Parameters for CWT
const char* wave = "morl"; // Morlet wavelet
float param = MORLET;         // Morlet parameter
int N = SAMPLE_LEN;              // Length of your signal
float dt = DT;          // Sampling rate (1 for example)
int J = SCALES;                // Total number of scales

float s0 = S0;
float dj = DJ;
float power = POW;
char * type = "pow";
float reference = REFERENCE;
cwt_object cwt_out;

void fail(void)
{
  printf("*** FAIL ***");
  while (1);
}

void load_input(uint32_t *user_input)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, user_input, 2500);
}

int check_output(void)
{
  int i;
  uint32_t mask, len;
  volatile uint32_t *addr;
  const uint32_t *ptr = sample_output;

  while ((addr = (volatile uint32_t *) *ptr++) != 0) {
    mask = *ptr++;
    len = *ptr++;
    for (i = 0; i < len; i++)
      if ((*addr++ & mask) != *ptr++) {
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.\n",
               i + 1, len, addr - 1, *(ptr - 1), *(addr - 1) & mask);
        return CNN_FAIL;
      }
  }

  return CNN_OK;
}

void cnn_inference(uint32_t * input)
{
    cnn_init(); // Bring state machine into consistent state
    cnn_load_weights(); // Load kernels
    cnn_load_bias(); // Not used in this network
    cnn_configure(); // Configure state machine
    load_input(input); // Load data input
    cnn_start(); // Start CNN processing

    while (cnn_time == 0)
    MXC_LP_EnterSleepMode(); // Wait for CNN
}

void print_sampleinput(void){
    printf("*** The sample input is: **** \n");
    //in chunks of 64
    for(int i = 0; i < INTERP_SIZE; i++){
        for(int j = 0; j < INTERP_SIZE; j++){
            printf("%0.2x", input_0[i*INTERP_SIZE + j]);
        }
        printf("\n");
    }
}


void get_user_input(uint8_t is_known_answer_test)
{
    fflush(stdin);
    fflush(stdout);

    //read in user input line by line from serial port
    char input[DEMO_INPUT_LEN*2];//DEMO_INPUT_LEN*2 because each char is 2 hex digits

    int expected_chunks = DEMO_INPUT_CHUNK_LEN;
    int i = 0;
    int total_input_i = 0;
    int chunk = 0;

    while (chunk < expected_chunks){
        printf("\nchunk(%d of %d):\n", chunk+1, expected_chunks);
        fflush(stdin);
        fflush(stdout);

        //read in one chunk at a time
        while(1){
            char c = getchar();
            if(i == DEMO_INPUT_CHUNK_LEN*2){
                //input[i] = '\0'; dont use the string terminator
                break;
            }
            if (c != '\n' && c != '\r'){
                input[total_input_i] = c;
                i++;
                total_input_i++;
            }
        }
        chunk++;
        i = 0;

        fflush(stdout);
        fflush(stdin);
    }

    printf("\n"); //do not remove this! it is needed to flush the serial buffer
    
    //convert chars to uint8_t
    int mismatch_count = 0;
    for(int j = 0; j < total_input_i; j+=2){
        char temp[2];
        uint8_t converted = 0; //important that his doesnt have type char. if it does, the matchings will be wrong
        temp[0] = input[j];
        temp[1] = input[j+1];
        converted = strtol(temp, NULL, 16);
        input_serial[j/2] = converted;

        //compare with input_0[i] if this is a known answer test
        if(is_known_answer_test && (input_serial[j/2] != input_0[j/2])){
            mismatch_count += 1;
        }
    }


    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_inference((uint32_t *)input_serial);
    cnn_unload((uint32_t *) cnn_output); //output should be 0x00007f7d
    cnn_disable(); // Shut down CNN clock, disable peripheral


    //is 0x7d for known answer test, first 8 bits of cnn_output
    uint8_t z1 = (uint8_t)(cnn_output[0] & 0x000000ff);

    //is 0x7f for known answer test
    uint8_t z2 = (uint8_t)((cnn_output[0] & 0x0000ff00) >> 8);

    if (is_known_answer_test){
        printf("Output: Comparing known input with serial input gave %d errors. CNN result: [%x, %x, %x] \n", mismatch_count, z1, z2, cnn_output);
    }else{
        printf("Output: %x, %x, %x\n", z1, z2, cnn_output);
    }


    //clear the input array
    for(int j = 0; j < DEMO_INPUT_LEN*2; j++){
        input[j] = '\0';
    }
}


void cwt_test(void){

    cwt_out = cwt_init(wave, param, N, dt, J);
    setCWTScales(cwt_out, s0, dj, type, power);
    cwt(cwt_out, mean_normal_signal); 


    // Computing and printing the magnitude of the CWT output and the scales
    printf("CWT Output NORMAL Magnitude:\n");
    double * magnitude_array = (double *) malloc(SAMPLE_LEN*SCALES* sizeof(double));


    for (int j = 0; j < cwt_out->J; j++) {  // Iterate over scales
        //printf("Scale %d: ", j);
        for (int i = 0; i < N; i++) {  // Iterate over signal length
            int index = j * N + i;  // Assuming output is a 1D array of size J * N
            magnitude_array[index] = sqrt(cwt_out->output[index].re * cwt_out->output[index].re +
                                    cwt_out->output[index].im * cwt_out->output[index].im);

            //PRINT DOUBLES
            printf("%lf ",  magnitude_array[index]);
        }
        //printf("\n");
    }
    printf("\n");


    free(magnitude_array);


    /* ####### FAULT SIGNAL  ################*/
    // Perform the Continuous Wavelet Transform
    cwt(cwt_out, mean_fault_signal); 

    magnitude_array = (double *) malloc(SAMPLE_LEN*SCALES* sizeof(double));

    printf("CWT Output FAULT Magnitude:\n");
    for (int j = 0; j < cwt_out->J; j++) {  // Iterate over scales
        //printf("Scale %d: ", j);
        for (int i = 0; i < N; i++) {  // Iterate over signal length
            int index = j * N + i;  // Assuming output is a 1D array of size J * N
            magnitude_array[index] = sqrt(cwt_out->output[index].re * cwt_out->output[index].re +
                                    cwt_out->output[index].im * cwt_out->output[index].im);

            //PRINT DOUBLES
            printf("%lf ",  magnitude_array[index]);

        }
        //printf("\n");
    }
    printf("\n");


    free(magnitude_array);



    double *scales = cwt_out->scale;         // Access the scales
    printf("\nScales:\n");
    for (int j = 0; j < cwt_out->J; j++) {
        printf("%f, ", cwt_out->scale[j]);
    }

}



void process_user_window_input(double *window){

    cwt_out = cwt_init(wave, param, N, dt, J);
    setCWTScales(cwt_out, s0, dj, type, power);
    cwt(cwt_out, window);

    fflush(stdin);
    printf("CWT Normalised: \n");
    fflush(stdout);
    
    for (int j = 0; j < cwt_out->J; j++) {  // Iterate over scales
        for (int i = 0; i < N; i++) {  // Iterate over signal length
            int index = j * N + i;  // Assuming output is a 1D array of size J * N
            double magnitude_normalised = sqrt(cwt_out->output[index].re * cwt_out->output[index].re +
                                    cwt_out->output[index].im * cwt_out->output[index].im) / REFERENCE;

            //PRINT DOUBLES
            printf("%lf ", magnitude_normalised);

        }
    }
    printf("\n");

}

void read_user_window_input(double *window){

    printf("Enter window: \n");

    fflush(stdin);
    fflush(stdout);

    for (int i = 0; i < SAMPLE_LEN; i += 1) {
        if (scanf("%lf", &window[i]) != 1) {
            // Handle error, e.g., not enough inputs
            break;
        }
    }

    fflush(stdin);
    printf("Window: \n");
    fflush(stdout);

    for(int i = 0; i < SAMPLE_LEN; i++){
        printf("%lf ", window[i]);
    }

}

void demo_main(void)
{   
    // ----------------- DEMO 1: DIRECT INPUT -----------------

    double window[SAMPLE_LEN];

    while (1){
        read_user_window_input(&window);
        process_user_window_input(&window);
    }

    
    // ----------------- DEMO 2: CWT IMG INPUT -----------------
    cwt_test();

    uint8_t is_known_answer_test = 1;
    while(1){
        get_user_input(is_known_answer_test);
        is_known_answer_test = 0;
    }
}

void demo_test_cnn(void)
{
    /*
    SUMMARY OF OPS
    Hardware: 14,050,098 ops (13,686,624 macc; 363,474 comp; 0 add; 0 mul; 0 bitwise)
        Layer 0: 1,600,000 ops (1,440,000 macc; 160,000 comp; 0 add; 0 mul; 0 bitwise)
        Layer 1: 11,700,000 ops (11,520,000 macc; 180,000 comp; 0 add; 0 mul; 0 bitwise)
        Layer 2: 684,288 ops (663,552 macc; 20,736 comp; 0 add; 0 mul; 0 bitwise)
        Layer 3: 64,944 ops (62,208 macc; 2,736 comp; 0 add; 0 mul; 0 bitwise)
        Layer 4: 866 ops (864 macc; 2 comp; 0 add; 0 mul; 0 bitwise)

    RESOURCE USAGE
    Weight memory: 13,104 bytes out of 442,368 bytes total (3.0%)
    Bias memory:   0 bytes out of 2,048 bytes total (0.0%)
    */

    if (!DEMO_ENABLED){
        printf("*** DEMO DISABLED ***");
        return;
    }else{
        printf("*** DEMO ENABLED ***");
        print_sampleinput();

        
    }
    printf("*** CNN Inference Test bearingnet ***");
    // Enable peripheral, enable CNN interrupt, turn on CNN clock
    // CNN clock: APB (50 MHz) div 1
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_inference((uint32_t *) input_0);

    if (check_output() != CNN_OK) fail();
    cnn_unload((uint32_t *) cnn_output);

    //clear output array
    for(int i = 0; i < (CNN_NUM_OUTPUTS + 3) / 4; i++){
        cnn_output[i] = 0;
    }

    printf("*** PASS ***");

    #ifdef CNN_INFERENCE_TIMER
        printf("Approximate inference time: %u us \n", cnn_time);
    #endif

    cnn_disable(); // Shut down CNN clock, disable peripheral
}