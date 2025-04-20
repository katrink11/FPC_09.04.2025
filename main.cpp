#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <complex>
#include <iostream>
#include <chrono>

const int WIDTH = 800;
const int HEIGHT = 800;
const int MAX_ITER = 100;

int mandelbrot(const std::complex<double>& c) {
    std::complex<double> z = 0;
    int iter = 0;
    while (std::abs(z) <= 2.0 && iter < MAX_ITER) {
        z = z * z + c;
        ++iter;
    }
    return iter;
}

cv::Vec3b getColor(int iter) {
    if (iter == MAX_ITER) {
        return { 0, 0, 0 };
    }
    int r = 9 * (255 - iter * 255 / MAX_ITER);
    int g = 15 * (255 - iter * 255 / MAX_ITER);
    int b = 8 * (255 - iter * 255 / MAX_ITER);
    return cv::Vec3b(b % 256, g % 256, r % 256);
}

void computeSequential(cv::Mat& image) {
    double xmin = -2.0, xmax = 1.0;
    double ymin = -1.5, ymax = 1.5;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double real = xmin + (x / static_cast<double>(WIDTH)) * (xmax - xmin);
            double imag = ymin + (y / static_cast<double>(HEIGHT)) * (ymax - ymin);
            std::complex<double> c(real, imag);
            int iter = mandelbrot(c);
            image.at<cv::Vec3b>(y, x) = getColor(iter);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    if (rank == 0) {
        
        cv::Mat seq_image(HEIGHT, WIDTH, CV_8UC3);
        auto start_seq = std::chrono::high_resolution_clock::now();

        computeSequential(seq_image);

        auto end_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_seq = end_seq - start_seq;
        std::cout << "[Sequential] Time: " << duration_seq.count() << " sec" << std::endl;

        
  
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    
    auto start_par = std::chrono::high_resolution_clock::now();

    int rows_per_proc = HEIGHT / size;
    int extra_rows = HEIGHT % size;
    int start_row = rank * rows_per_proc + std::min(rank, extra_rows);
    int local_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);

    cv::Mat local_image(local_rows, WIDTH, CV_8UC3);

    double xmin = -2.0, xmax = 1.0;
    double ymin = -1.5, ymax = 1.5;

    for (int y = 0; y < local_rows; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            double real = xmin + (x / static_cast<double>(WIDTH)) * (xmax - xmin);
            double imag = ymin + ((start_row + y) / static_cast<double>(HEIGHT)) * (ymax - ymin);
            std::complex<double> c(real, imag);
            int iter = mandelbrot(c);
            local_image.at<cv::Vec3b>(y, x) = getColor(iter);
        }
    }

    cv::Mat full_image;
    if (rank == 0) {
        full_image = cv::Mat(HEIGHT, WIDTH, CV_8UC3);
    }

    int* recvcounts = nullptr;
    int* displs = nullptr;
    if (rank == 0) {
        recvcounts = new int[size];
        displs = new int[size];
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int rows = rows_per_proc + (i < extra_rows ? 1 : 0);
            recvcounts[i] = rows * WIDTH * 3;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    MPI_Gatherv(local_image.data, local_rows * WIDTH * 3, MPI_UNSIGNED_CHAR,
        full_image.data, recvcounts, displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_par = end_par - start_par;

    if (rank == 0) {
        std::cout << "[Parallel (MPI)] Time: " << duration_par.count() << " sec" << std::endl;
        cv::imshow("Parallel Mandelbrot", full_image);
        cv::waitKey(0);
        delete[] recvcounts;
        delete[] displs;
    }

    MPI_Finalize();
    return 0;
}
