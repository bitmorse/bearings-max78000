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

#define DEMO_INPUT_LEN 64*64
#define DEMO_INPUT_CHUNK_LEN 64

#define INTERPOLATION_W 48
#define INTERPOLATION_H 48
#define SAMPLE_LENGTH 192
#define SCALES 12
#define MORLET 5

#define PI 3.14159265358979323846

volatile uint32_t cnn_time; // Stopwatch

// Expected output of layer 4 for bearingnet given the sample input (known-answer test)
static const uint32_t sample_output[] = SAMPLE_OUTPUT; 

// 1-channel 64x64 data input (4096 bytes / 1024 32-bit words):
// HWC 64x64, channels 0 to 0
static const uint32_t input_0[] = SAMPLE_INPUT_0;

static int32_t  cnn_output[(CNN_NUM_OUTPUTS + 3) / 4]; // CNN output data

uint32_t input_serial[DEMO_INPUT_LEN]; //the input from the serial port by the user

void fail(void)
{
  printf("*** FAIL ***");
  while (1);
}

void load_input(uint32_t *user_input)
{
  // This function loads the sample data input -- replace with actual data

  memcpy32((uint32_t *) 0x50400000, user_input, 4096);
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
        printf("Data mismatch (%d/%d) at address 0x%08x: Expected 0x%08x, read 0x%08x.",
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
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 64; j++){
            printf("%0.2x", input_0[i*64 + j]);
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

//input_image (J rows, N columns) 
void call_bilinear_interpolate(double *input_image, double *input_image_interpolated, int N, int J, int new_width, int new_height) {
    for (int i = 0; i < new_width; i++) {
        for (int j = 0; j < new_height; j++) {
            // Adjusted calculations to ensure coordinates are within bounds
            double x = (i / (double)(new_width - 1)) * (N - 1);
            double y = (j / (double)(new_height - 1)) * (J - 1);

            int x1 = (int) x;
            int y1 = (int) y;
            int x2 = fmin(x1 + 1, N);
            int y2 = fmin(y1 + 1, J);

            double wa = (x2 - x) * (y2 - y);
            double wb = (x - x1) * (y2 - y);
            double wc = (x2 - x) * (y - y1);
            double wd = (x - x1) * (y - y1);

            int index = i * new_height + j;
            int input_index1 = x1 * J + y1;
            int input_index2 = x2 * J + y1;
            int input_index3 = x1 * J + y2;
            int input_index4 = x2 * J + y2;

            input_image_interpolated[index] = wa * input_image[input_index1] +
                                              wb * input_image[input_index2] +
                                              wc * input_image[input_index3] +
                                              wd * input_image[input_index4];
        }
    }
}

void cwt_test(void){



    //double signal[SAMPLE_LENGTH];
    //double frequency = 5;
    //double sampling_rate = SAMPLE_LENGTH; // Assuming 1 second duration
    //double t;

    // Generate a test signal
    /*
    for (int i = 0; i < SAMPLE_LENGTH; i++) {
        t = (double)i / sampling_rate;
        signal[i] = sin(2 * PI * frequency * t);
    }
    */

    // Parameters for CWT
    const char* wave = "morl"; // Morlet wavelet
    float param = MORLET;         // Morlet parameter
    int N = SAMPLE_LENGTH;              // Length of your signal
    float dt = 1;          // Sampling rate (1 for example)
    int J = SCALES;                // Total number of scales



    // Initialize the CWT object
    cwt_object obj = cwt_init(wave, param, N, dt, J);

    printf("J: %i\n", obj->J);
    printf("dj: %f\n", obj->dj);
    printf("scale type: %s\n", obj->type);

    // Perform the Continuous Wavelet Transform
    cwt(obj, mean_normal_signal); 

    //printf("signal: \n");
    //for(int i = 0; i < SAMPLE_LENGTH; i++){
    //    printf("%f ", signal[i]);
    //}

        // Computing and printing the magnitude of the CWT output and the scales
    printf("CWT Output NORMAL Magnitude:\n");
    double * magnitude_array = (double *) malloc(SAMPLE_LENGTH*SCALES* sizeof(double));
    double * magnitude_array_interpolated = (double *) malloc(INTERPOLATION_W*INTERPOLATION_H* sizeof(double));


    for (int j = 0; j < obj->J; j++) {  // Iterate over scales
        //printf("Scale %d: ", j);
        for (int i = 0; i < N; i++) {  // Iterate over signal length
            int index = j * N + i;  // Assuming output is a 1D array of size J * N
            magnitude_array[index] = sqrt(obj->output[index].re * obj->output[index].re +
                                    obj->output[index].im * obj->output[index].im);

            //PRINT DOUBLES
            printf("%lf ",  magnitude_array[index]);
        }
        //printf("\n");
    }
    printf("\n");

    call_bilinear_interpolate(magnitude_array, magnitude_array_interpolated, N, J, INTERPOLATION_W, INTERPOLATION_H);

    printf("CWT Output NORMAL Interpolation:\n");
    for (int i = 0; i < INTERPOLATION_W*INTERPOLATION_H; i++) {  // Iterate over scales
        printf("%lf ",  magnitude_array_interpolated[i]);
    }


    free(magnitude_array);
    free(magnitude_array_interpolated);



    /* ####### FAULT SIGNAL  ################*/
    // Perform the Continuous Wavelet Transform
    cwt(obj, mean_fault_signal); 

    magnitude_array = (double *) malloc(SAMPLE_LENGTH*SCALES* sizeof(double));
    magnitude_array_interpolated = (double *) malloc(INTERPOLATION_W*INTERPOLATION_H* sizeof(double));

    printf("CWT Output FAULT Magnitude:\n");
    for (int j = 0; j < obj->J; j++) {  // Iterate over scales
        //printf("Scale %d: ", j);
        for (int i = 0; i < N; i++) {  // Iterate over signal length
            int index = j * N + i;  // Assuming output is a 1D array of size J * N
            magnitude_array[index] = sqrt(obj->output[index].re * obj->output[index].re +
                                    obj->output[index].im * obj->output[index].im);

            //PRINT DOUBLES
            printf("%lf ",  magnitude_array[index]);

        }
        //printf("\n");
    }
    printf("\n");


    call_bilinear_interpolate(magnitude_array, magnitude_array_interpolated, N, J, INTERPOLATION_W, INTERPOLATION_H);

    printf("CWT Output FAULT Interpolation:\n");

    for (int i = 0; i < INTERPOLATION_W*INTERPOLATION_H; i++) { 
        printf("%lf ",  magnitude_array_interpolated[i]);
    }

    free(magnitude_array);
    free(magnitude_array_interpolated);



    double *scales = obj->scale;         // Access the scales
    printf("\nScales:\n");
    for (int j = 0; j < obj->J; j++) {
        printf("%f, ", obj->scale[j]);
    }


    // Now signal[] contains the test signal similar to the generated one
    // ... (rest of your code)
    


}

void demo_main(void)
{
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
    Hardware: 23,124,226 ops (22,525,440 macc; 598,786 comp; 0 add; 0 mul; 0 bitwise)
        Layer 0: 2,621,440 ops (2,359,296 macc; 262,144 comp; 0 add; 0 mul; 0 bitwise)
        Layer 1: 19,169,280 ops (18,874,368 macc; 294,912 comp; 0 add; 0 mul; 0 bitwise)
        Layer 2: 1,216,512 ops (1,179,648 macc; 36,864 comp; 0 add; 0 mul; 0 bitwise)
        Layer 3: 115,456 ops (110,592 macc; 4,864 comp; 0 add; 0 mul; 0 bitwise)
        Layer 4: 1,538 ops (1,536 macc; 2 comp; 0 add; 0 mul; 0 bitwise)

    RESOURCE USAGE
    Weight memory: 26,880 bytes out of 442,368 bytes total (6.1%)
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