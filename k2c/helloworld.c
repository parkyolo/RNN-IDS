#include <stdio.h> 
#include <math.h> 
#include <time.h> 
#include <string.h> 
#include <stdlib.h>

int main() {
    int max = 1024;
    FILE *f = fopen("simpleRNNIDS_train.csv", "r");
    char line[max];
    char *pLine;
    while(!feof(f)) {
        pLine = fgets(line, max, f);
        // printf("%s\n",pLine);
        // // char lline[max];
        // // pLine = fgets(lline, max, f);
        // // printf("%s\n", pLine);
        // char *ptr = strtok(pLine, ",");
        // float simple_rnn_input_input_array[122] = {0.0};
        // int i = 0;
        // while (ptr != NULL) {
        //     if (i == 122) break;
        //     simple_rnn_input_input_array[i] = atof(ptr);
        //     ptr = strtok(NULL, ",");
        //     i++;
        // }

        // printf("%f", simple_rnn_input_input_array[19]);
    }
    
}
