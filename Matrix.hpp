// Matrix Class
// Ben Crabtree, 2021

# ifndef __MATRIX_HPP
# define __MATRIX_HPP

# include <vector>
# include <string>
# include <utility> // std::pair

class Matrix
{
    int Rows;
    int Cols;
    std::vector<std::vector<float>> Vals;
    bool Augmented;
    bool REF;

    public:
    // Default Constructor
    Matrix();
    // Constructor for zero matrix, matrix of 1s or identity matrix
    Matrix(int rows, int cols, int matrixType);
    // General Constructor
    Matrix(int rows, int cols, std::vector<std::vector<float>> vals);
    // Destructor
    ~Matrix();

    // Methods
    int get_Rows();
    int get_Cols();
    std::vector<std::vector<float>> get_Vals();
    void printMatrix();
    void printDims();
    void transpose();
    void setAttributes(int rows, int cols, std::vector<std::vector<float>> vals);
    // l_pq matrix norm
    float lpqNorm(float p, float q);
    
    // Overloading Operators
    friend Matrix operator + (Matrix const & M, Matrix const & N);
    friend Matrix operator - (Matrix const & M, Matrix const & N);
    // scalar multiplication (overload *)
    friend Matrix operator * (float const & a, Matrix const & M);
    // matrix multiplication (overload *)
    friend Matrix operator * (Matrix const & M, Matrix const & N);

    //Matrix dot_product(Matrix M, Matrix N);

    // row operations
    // 1. Row addition - Adds row2 to row1 (modifies row1)
    void rowAdd(int row1, int row2);
    // 2. Row scaling - multiplying all entries of a row by a non-zero constant c.
    void rowScale(int row, float c);
    // 3. Row swapping - interchanging two rows of a matrix.
    void rowSwap(int row1, int row2);

    // Augment matrix with b column to solve eqn A*x = b
    void augment(std::vector<float> b);
    // Just in case
    void deaugment();
    // Gaussian elimination to put matrix in row eschelon form
    float gaussElim();
    // solve() augments matrix and puts in REF
    // then performs back substitution to solve eqn A*x = b for vector x,
    // returned as a nx1 matrix object (column vector)
    Matrix solve(std::vector<float> b);
    
    // FOR SQUARE MATRICES:
    // Trace
    float trace();
    // Determinant
    float determinant();

    // Gram Schmidt Process
    std::vector<std::vector<float>> gram_schmidt();
    // QR Decomposition - returns a pair of matrices, a Q and R
    std::pair<Matrix, Matrix> qr_decomp();
    // QR Algorithm
    // Returns a pair containing a vector and a matrix
    // The elements of the vector are the eigenvalues
    // The columns of the matrix are the associated eigenvectors
    std::pair<std::vector<float>, Matrix> qr_algo(int iterations);
    
    // TO DO:
    // Inverse?
    Matrix inverse();
    // Orthogonal (returns bool. A matrix A is orthogonal if its transpose is equal to its inverse)
    bool orthogonal();

};

// Overload operator function declarations
Matrix operator + (Matrix const & M, Matrix const & N);
Matrix operator * (float const & a, Matrix const & M);
Matrix operator * (Matrix const & M, Matrix const & N);
Matrix operator - (Matrix const & M, Matrix const & N);

# endif