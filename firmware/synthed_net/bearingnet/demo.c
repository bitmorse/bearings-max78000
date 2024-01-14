#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "demo.h"
#include "mxc.h"
#include "cnn.h"
#include "sampledata.h"
#include "sampleoutput.h"

#define DEMO_INPUT_LEN 64*64
#define DEMO_INPUT_CHUNK_LEN 64

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

    uint8_t z1 = (uint8_t)cnn_output[0]; //is 0x7d
    uint8_t z2 = (uint8_t)(cnn_output[0] >> 8); //is 0x7f

    if (is_known_answer_test){
        printf("Output: Comparing known input with serial input gave %d errors. CNN result: [%x, %x] \n", mismatch_count, z1, z2);
    }else{
        printf("Output: %x, %x\n", mismatch_count, z1, z2);
    }


    //clear the input array
    for(int j = 0; j < DEMO_INPUT_LEN*2; j++){
        input[j] = '\0';
    }
}


void demo_main(void)
{
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

    printf("*** PASS ***");

    #ifdef CNN_INFERENCE_TIMER
        printf("Approximate inference time: %u us \n", cnn_time);
    #endif

    cnn_disable(); // Shut down CNN clock, disable peripheral
}