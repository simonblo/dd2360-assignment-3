#include "Timing.h"

double cpuSecond()
{
    struct timespec tp;
    timespec_get(&tp, TIME_UTC);
    return ((double)tp.tv_sec + (double)tp.tv_nsec*1.e-9);
}
