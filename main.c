#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>

#define TRUE_PI 3.14159265359
#define NUM_ITERATIONS 3

struct thread_data {
    long long points_to_generate;
    unsigned int seed[4];
    long long local_inside;
    int use_shared_counter;
};

pthread_mutex_t mutex;
long long shared_inside;

double rand_double(unsigned int state[4]) {
    uint64_t x = ((uint64_t)state[0] << 32) | state[1];
    uint64_t y = ((uint64_t)state[2] << 32) | state[3];
    
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    
    state[0] = (unsigned int)(y >> 32);
    state[1] = (unsigned int)y;
    state[2] = (unsigned int)(x >> 32);
    state[3] = (unsigned int)x;
    
    return (x * (1.0 / 18446744073709551616.0));
}

void* thread_func(void* arg) {
    struct thread_data* data = (struct thread_data*)arg;
    long long local_inside = 0;
    for (long long i = 0; i < data->points_to_generate; i++) {
        double x = rand_double(data->seed) * 2 - 1;
        double y = rand_double(data->seed) * 2 - 1;
        if (x * x + y * y <= 1) {
            if (data->use_shared_counter) {
                pthread_mutex_lock(&mutex);
                shared_inside++;
                pthread_mutex_unlock(&mutex);
            } else {
                local_inside++;
            }
        }
    }
    if (!data->use_shared_counter) {
        data->local_inside = local_inside;
    }
    return NULL;
}

double approach1(long long nPoints, double* timeTaken) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    long long inside = 0;
    
    unsigned int state[4];
    state[0] = (unsigned int)time(NULL);
    state[1] = (unsigned int)(time(NULL) >> 32);
    state[2] = (unsigned int)clock();
    state[3] = (unsigned int)(clock() << 16);
    
    for (long long i = 0; i < nPoints; i++) {
        double x = rand_double(state) * 2 - 1;
        double y = rand_double(state) * 2 - 1;
        if (x * x + y * y <= 1) inside++;
    }
    double pi = 4.0 * inside / nPoints;
    clock_gettime(CLOCK_MONOTONIC, &end);
    *timeTaken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return pi;
}

double multi_threaded_approach(long long nPoints, int nThreads, int use_shared_counter, double* timeTaken) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    long long base_points = nPoints / nThreads;
    long long extra = nPoints % nThreads;
    struct thread_data* data = malloc(nThreads * sizeof(struct thread_data));
    
    uint64_t base_seed = (uint64_t)time(NULL);
    
    for (int i = 0; i < nThreads; i++) {
        data[i].points_to_generate = base_points + (i < extra ? 1 : 0);
        data[i].seed[0] = (unsigned int)base_seed + i * 1327;
        data[i].seed[1] = (unsigned int)(base_seed >> 32) + i * 7919;
        data[i].seed[2] = (unsigned int)(base_seed * 104729) + i * 293;
        data[i].seed[3] = (unsigned int)((base_seed * 104729) >> 32) + i * 9781;
        data[i].local_inside = 0;
        data[i].use_shared_counter = use_shared_counter;
    }
    
    shared_inside = 0;
    pthread_t* threads = malloc(nThreads * sizeof(pthread_t));
    for (int i = 0; i < nThreads; i++) {
        pthread_create(&threads[i], NULL, thread_func, &data[i]);
    }
    for (int i = 0; i < nThreads; i++) {
        pthread_join(threads[i], NULL);
    }
    long long total_inside;
    if (use_shared_counter) {
        total_inside = shared_inside;
    } else {
        total_inside = 0;
        for (int i = 0; i < nThreads; i++) {
            total_inside += data[i].local_inside;
        }
    }
    double pi = 4.0 * total_inside / nPoints;
    clock_gettime(CLOCK_MONOTONIC, &end);
    *timeTaken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    free(data);
    free(threads);
    return pi;
}

int main() {
    pthread_mutex_init(&mutex, NULL);
    long long nPoints_array[] = {1000000, 10000000, 100000000};
    int nThreads_array[] = {1, 2, 4, 8, 16, 32, 64, 100, 200, 500, 1000};
    int nThreads_count = sizeof(nThreads_array) / sizeof(nThreads_array[0]);

    printf("iteration,approach,nPoints,nThreads,time,pi,error\n");

    // Approach 1: Single-threaded
    for (int i = 0; i < 3; i++) {
        long long nPoints = nPoints_array[i];
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            double timeTaken;
            double pi = approach1(nPoints, &timeTaken);
            double error = fabs(pi - TRUE_PI);
            printf("%d,1,%lld,1,%f,%f,%f\n", iter + 1, nPoints, timeTaken, pi, error);
        }
    }

    // Approach 2 and 3: Multi-threaded with local and shared counters
    for (int i = 0; i < 3; i++) {
        long long nPoints = nPoints_array[i];
        for (int j = 0; j < nThreads_count; j++) {
            int nThreads = nThreads_array[j];
            for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
                double timeTaken;

                // Approach 2: Local counters
                double pi = multi_threaded_approach(nPoints, nThreads, 0, &timeTaken);
                double error = fabs(pi - TRUE_PI);
                printf("%d,2,%lld,%d,%f,%f,%f\n", iter + 1, nPoints, nThreads, timeTaken, pi, error);

                // Approach 3: Shared counter with mutex
                pi = multi_threaded_approach(nPoints, nThreads, 1, &timeTaken);
                error = fabs(pi - TRUE_PI);
                printf("%d,3,%lld,%d,%f,%f,%f\n", iter + 1, nPoints, nThreads, timeTaken, pi, error);
            }
        }
    }

    pthread_mutex_destroy(&mutex);
    return 0;
}