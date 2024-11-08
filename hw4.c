/*
 * Game of Life MPI Implementation
 * Adapted from Anderson B. Liddle's OpenMP implementation
 * To Compile: mpicc -Wall -O3 -o anderson_mpi anderson_mpi.c
 * To Run: mpiexec -n <num_threads> ./anderson_mpi <dimensions> <MAX_GEN> <num_threads> <output_directory> <stagnationcheck>
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <mpi.h>


// Dynamically allocate a 2D array of integers
int **allocarray(int rows, int cols) {
    int* data = (int*)malloc(rows * cols * sizeof(int));
    int** array = (int**)malloc(rows * sizeof(int*));

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

// Randomizes the grid with 0s or 1s to create a random initial state
int** genzero(int** array, int rows, int cols){
  srand(42);  // Fixed seed for reproducibility
  //unsigned int seed = 1;
  int i,j;

  for (i = 1; i < rows - 1; i++)
    for (j = 1; j < cols - 1; j++)
      array[i][j] = rand() % 2;
  
  return array;
}

// Count neighbors and update cell state
int compute_next_state(int** lastgrid, int i, int j) {
    // Sum all 8 neighboring values to check if cell should live or die
    int neighbors = lastgrid[i - 1][j - 1]
                + lastgrid[i - 1][j]
                + lastgrid[i - 1][j + 1]
                + lastgrid[i][j + 1]
                + lastgrid[i][j - 1]
                + lastgrid[i + 1][j]
                + lastgrid[i + 1][j - 1]
                + lastgrid[i + 1][j + 1];
      if (neighbors <= 1){ // Dies of starvation
        return 0;
      }
      else if (neighbors >= 4){ // Dies of overpopulation
        return 0;
      }
      else if ((neighbors == 3) && (lastgrid[i][j] == 0)){ // Dead cells is born again
        return 1;
      }
      else if ((2 <= neighbors) && (neighbors <= 3) && (lastgrid[i][j] == 1)){ // Alive cell remains 1
        return 1;
      }
      else{ // Every other cell remains 0
        return 0;
      }
      return 0; // Just in case
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
void outputtofile(char *output_file, int** grid, int rows, int cols){
    FILE *file = fopen(output_file, "w");
    int i,j;
  
    for (i = 1; i < rows - 1; i++) {
        for (j = 1; j < cols - 1; j++)
          if (grid[i][j]){
            fprintf(file, "%i ",grid[i][j]);
          }
          else{
            fprintf(file, "%i ", grid[i][j]);
          }
        fprintf(file, "\n");
      }
      fclose(file);
}

// Function based on code provided in matmul.c
// Gets the time and is used for benchmarking
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command line arguments
    if (argc != 6) {
        if (rank == 0) {
            printf("Usage: %s <dimensions> <MAX_GEN> <num_threads> <output_directory> <stagnationcheck>\n", 
                   argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Getting the command line arguments
    int dimensions      = atoi(argv[1]);
    int MAX_GEN         = atoi(argv[2]);

    // Getting number of threads
    int num_threads     = atoi(argv[3]);
    
    // Boolean for turning on and off stagnation check
    int stagnationcheck = atoi(argv[5]);

    // Output file and directory (format output_N_N_gen_threads.txt)
    char output_file[200];
    sprintf(output_file, "%s/output%s_%s_%s.txt", argv[4], argv[1], argv[2], argv[3]);

    // Verify number of processes
    if (num_threads != size) {
        if (rank == 0) {
            printf("Error: Number of processes (%d) does not match MPI size (%d)\n", 
                   num_threads, size);
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate local dimensions
    int rows_per_proc   = dimensions / size;
    int remainder       = dimensions % size;
    int local_rows      = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate arrays for scatter/gather operations
    int* sendcounts     = (int*)malloc(size * sizeof(int));
    int* displs         = (int*)malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (dimensions / size + (i < remainder ? 1 : 0)) * dimensions;
        displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
    }

    // Allocate local grids with ghost rows
    int total_local_rows    = local_rows + 2;  // Add 2 for ghost rows
    int total_cols          = dimensions + 2;        // Add 2 for ghost columns
    int** current           = allocarray(total_local_rows, total_cols);
    int** next              = allocarray(total_local_rows, total_cols);
    
    initarray(current, total_local_rows, total_cols);
    initarray(next, total_local_rows, total_cols);

    // Process 0 initializes and distributes the grid
    int* global_grid = NULL;
    if (rank == 0) {
        // Making global grid
        global_grid     = (int*)malloc(dimensions * dimensions * sizeof(int));
        int** temp_grid = allocarray(dimensions + 2, dimensions + 2);
        
        // Initializing the grid with a random board state
        initarray(temp_grid, dimensions + 2, dimensions + 2);
        genzero(temp_grid, dimensions + 2, dimensions + 2);
        
        // Copy to linear array for scattering
        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                global_grid[i * dimensions + j] = temp_grid[i + 1][j + 1];
        
        destroyarray(temp_grid);
    }

    // Allocate buffer for local section
    int* local_grid = (int*)malloc(sendcounts[rank] * sizeof(int));

    // Scatter the initial grid
    MPI_Scatterv(global_grid, sendcounts, displs, MPI_INT,
                 local_grid, sendcounts[rank], MPI_INT,
                 0, MPI_COMM_WORLD);

    // Copy received data to local grid
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < dimensions; j++)
            current[i + 1][j + 1] = local_grid[i * dimensions + j];

    // Process 0 starts timer
    double starttime;
    if (rank == 0){
        starttime = gettime();
    }
    
    // Generation counter
    int gen;

    // Flag to store whether the board state has changed or not
    int global_changed = 1;
    
    // Iteration loop for main logic. Endes after required number of generations
    for (gen = 1; gen <= MAX_GEN; gen++) {
        // Exchange ghost rows with neighboring processes, passing both upwards and downwards
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

        // Updating grid based on cell values
        // Process local section
        process_section(next, current, 1, local_rows + 1, total_cols);

        // Checking for stagnation and breaking loop if grid has not changed
        // Checks stagnationcheck boolean first to ensure function is not run if false
        // Saves some time
        int local_changed = 0;
        if (stagnationcheck) {
            local_changed = section_changed(next, current, 1, local_rows + 1, total_cols);
            MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (!global_changed){
                printf("Breaking at generation %i\n", gen);
                break;
            }
        }

        // Swapping grid pointers to make current grid the lastgrid
        // Simply assigns current grid as old last rid, as all values will be updated. No need to clear values.
        int** temp  = current;
        current     = next;
        next        = temp;
    }

    // Pack local data for gathering
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < dimensions; j++)
            local_grid[i * dimensions + j] = current[i + 1][j + 1];

    // Gather final state
    if (rank == 0) {
        if (global_grid == NULL)
            global_grid = (int*)malloc(dimensions * dimensions * sizeof(int));
    }

    // Gathering all final states from among the processes
    MPI_Gatherv(local_grid, sendcounts[rank], MPI_INT,
                global_grid, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    // Process 0 makes output file and calculates timing benchmark
    if (rank == 0) {
        // Creating array to hold final grid
        int** final_grid = allocarray(dimensions + 2, dimensions + 2);
        initarray(final_grid, dimensions + 2, dimensions + 2);
        
        // Collecting final grid
        for (int i = 0; i < dimensions; i++)
            for (int j = 0; j < dimensions; j++)
                final_grid[i + 1][j + 1] = global_grid[i * dimensions + j];

        // Outputting final board state to file
        outputtofile(output_file, final_grid, dimensions + 2, dimensions + 2);

        // Getting endtime and getting benchmarks
        double endtime = gettime();
        printf("Time taken = %lf seconds\n", endtime-starttime);

        // Freeing arrays
        destroyarray(final_grid);
        free(global_grid);
    }

    // Freeing arrays
    destroyarray(current);
    destroyarray(next);

    // Freeing helper variables
    free(sendcounts);
    free(displs);
    free(local_grid);

    MPI_Finalize();
    return 0;
}