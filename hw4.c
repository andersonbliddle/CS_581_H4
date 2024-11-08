/*
 * Game of Life MPI Implementation
 * Adapted from Anderson B. Liddle's OpenMP implementation
 * To Compile: mpicc -Wall -O3 -o anderson_mpi anderson_mpi.c
 * To Run: mpiexec -n <num_processes> ./anderson_mpi <dimensions> <max_generations> <num_processes> <output_directory> <stagnationcheck>
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

// Function to get current time
double gettime(void) {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec/1000000.0;
}

// Dynamically allocate a 2D array of integers
int **allocarray(int rows, int cols) {
    int* data = (int*)malloc(rows * cols * sizeof(int));
    int** array = (int**)malloc(rows * sizeof(int*));
    
    if (data == NULL || array == NULL) {
        printf("Error allocating memory\n");
        return NULL;
    }

    // For row major storage
    for (int i = 0; i < rows; i++)
        array[i] = &(data[cols * i]);
    
    return array;
}

// Free allocated memory
void destroyarray(int** array) {
    free(*array);  // Free the data
    free(array);   // Free the pointers
}

// Initialize array with zeros
void initarray(int **a, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            a[i][j] = 0;
}

// Initialize random board state
void genzero(int **array, int rows, int cols) {
    srand(42);  // Fixed seed for reproducibility
    for (int i = 1; i < rows - 1; i++)
        for (int j = 1; j < cols - 1; j++)
            array[i][j] = rand() % 2;
}

// Count neighbors and update cell state
int compute_next_state(int** lastgrid, int i, int j) {
    int neighbors = lastgrid[i - 1][j - 1] + lastgrid[i - 1][j] + lastgrid[i - 1][j + 1] +
                   lastgrid[i][j - 1] + lastgrid[i][j + 1] +
                   lastgrid[i + 1][j - 1] + lastgrid[i + 1][j] + lastgrid[i + 1][j + 1];
    
    if (neighbors <= 1) return 0;  // Dies of starvation
    if (neighbors >= 4) return 0;  // Dies of overpopulation
    if (neighbors == 3) return 1;  // Birth or survival
    if (neighbors == 2) return lastgrid[i][j];  // Survival if already alive
    return 0;  // Default case
}

// Process a section of the grid
void process_section(int** grid, int** lastgrid, int start_row, int end_row, int cols) {
    for (int i = start_row; i < end_row; i++)
        for (int j = 1; j < cols - 1; j++)
            grid[i][j] = compute_next_state(lastgrid, i, j);
}

// Check if section has changed
int section_changed(int** grid, int** lastgrid, int start_row, int end_row, int cols) {
    for (int i = start_row; i < end_row; i++)
        for (int j = 1; j < cols - 1; j++)
            if (grid[i][j] != lastgrid[i][j])
                return 1;
    return 0;
}

// Write final state to file
void write_to_file(char* output_dir, int** grid, int rows, int cols, 
                   int num_processes, int dimensions, int max_generations) {
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s/final_board_p%d_s%d_g%d.txt", 
             output_dir, num_processes, dimensions, max_generations);
    
    FILE* fp = fopen(filepath, "w");
    if (fp == NULL) {
        printf("Error opening output file %s\n", filepath);
        return;
    }

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            fprintf(fp, "%d ", grid[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command line arguments
    if (argc != 6) {
        if (rank == 0) {
            printf("Usage: %s <dimensions> <max_generations> <num_processes> <output_directory> <stagnationcheck>\n", 
                   argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Parse arguments
    int dimensions = atoi(argv[1]);
    int max_generations = atoi(argv[2]);
    int num_processes = atoi(argv[3]);
    char* output_dir = argv[4];
    int stagnationcheck = atoi(argv[5]);

    // Verify number of processes
    if (num_processes != size) {
        if (rank == 0) {
            printf("Error: Number of processes (%d) does not match MPI size (%d)\n", 
                   num_processes, size);
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate local dimensions
    int rows_per_proc = dimensions / size;
    int remainder = dimensions % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate arrays for scatter/gather operations
    int* sendcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (dimensions / size + (i < remainder ? 1 : 0)) * dimensions;
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }

    // Allocate local boards with ghost rows
    int total_local_rows = local_rows + 2;  // Add 2 for ghost rows
    int total_cols = dimensions + 2;        // Add 2 for ghost columns
    int** current = allocarray(total_local_rows, total_cols);
    int** next = allocarray(total_local_rows, total_cols);
    
    initarray(current, total_local_rows, total_cols);
    initarray(next, total_local_rows, total_cols);

    // Process 0 initializes and distributes the board
    int* global_board = NULL;
    if (rank == 0) {
        global_board = (int*)malloc(dimensions * dimensions * sizeof(int));
        int** temp_board = allocarray(dimensions + 2, dimensions + 2);
        initarray(temp_board, dimensions + 2, dimensions + 2);
        genzero(temp_board, dimensions + 2, dimensions + 2);
        
        // Copy to linear array for scattering
        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                global_board[i * dimensions + j] = temp_board[i + 1][j + 1];
        
        destroyarray(temp_board);
    }

    // Allocate buffer for local section
    int* local_board = (int*)malloc(sendcounts[rank] * sizeof(int));

    // Scatter the initial board
    MPI_Scatterv(global_board, sendcounts, displs, MPI_INT,
                 local_board, sendcounts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    // Copy received data to local board
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < dimensions; j++)
            current[i + 1][j + 1] = local_board[i * dimensions + j];

    // Start timer
    double start_time = MPI_Wtime();
    
    // Main game loop
    int generation;
    int global_changed = 1;
    
    for (generation = 0; generation < max_generations && (global_changed || !stagnationcheck); generation++) {
        // Exchange ghost rows
        if (rank > 0) {
            MPI_Sendrecv(current[1], total_cols, MPI_INT, rank - 1, 0,
                        current[0], total_cols, MPI_INT, rank - 1, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(current[local_rows], total_cols, MPI_INT, rank + 1, 1,
                        current[local_rows + 1], total_cols, MPI_INT, rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Process local section
        process_section(next, current, 1, local_rows + 1, total_cols);

        // Check for changes if stagnation check is enabled
        int local_changed = 0;
        if (stagnationcheck) {
            local_changed = section_changed(next, current, 1, local_rows + 1, total_cols);
            MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }

        // Swap current and next boards
        int** temp = current;
        current = next;
        next = temp;
    }

    // Pack local data for gathering
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < dimensions; j++)
            local_board[i * dimensions + j] = current[i + 1][j + 1];

    // Gather final state
    if (rank == 0) {
        if (global_board == NULL)
            global_board = (int*)malloc(dimensions * dimensions * sizeof(int));
    }

    MPI_Gatherv(local_board, sendcounts[rank], MPI_INT,
                global_board, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Process 0 writes final state and prints timing information
    if (rank == 0) {
        int** final_board = allocarray(dimensions + 2, dimensions + 2);
        initarray(final_board, dimensions + 2, dimensions + 2);
        
        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                final_board[i + 1][j + 1] = global_board[i * dimensions + j];

        write_to_file(output_dir, final_board, dimensions + 2, dimensions + 2,
                     num_processes, dimensions, max_generations);

        double end_time = MPI_Wtime();
        printf("Game finished after %d generations\n", generation);
        printf("Time taken: %f seconds\n", end_time - start_time);

        destroyarray(final_board);
        free(global_board);
    }

    // Cleanup
    destroyarray(current);
    destroyarray(next);
    free(sendcounts);
    free(displs);
    free(local_board);

    MPI_Finalize();
    return 0;
}